# Requirements Document

## Introduction

This document specifies requirements for the Full-Stack Agent Orchestration System — an extension of the existing FastAPI-based chatapp backend that introduces a Planner Agent layer, real-time data infrastructure (Redis, Kafka), service-oriented tools (User Service, Task Service), semantic memory via pgvector, WebSocket-based real-time chat, a unified Agent Gateway entry point, an intelligent Tool Router, an agent-callable Real-Time Events tool for live state and event data, a tool-mediated data-access security invariant, and live-data freshness guarantees. The system builds upon the already-implemented Tool Registry, Tool Executor, Decision Engine, Token Budget Manager, Citation Tracker, and multi-provider LLM fallback router, extending the architecture into a full production-grade agent platform.

The end-to-end reasoning flow this spec realizes is: the Agent_Gateway receives a user message → the Decision_Engine detects intent and decides whether live data is required → for simple requests it answers directly, and for compound requests the Planner_Agent produces an Execution_Plan → the Tool_Router dispatches each Tool_Call to the correct controlled tool (service, database, memory, real-time events, or external) → live data is fetched from the source of truth → the LLM_Provider reasons over the fresh results and produces the final response or action → real-time layers (Redis, Kafka, WebSocket, FCM) stream state, events, and notifications throughout. Consistent with the reference architecture, the LLM never accesses databases or infrastructure directly; every real-system interaction is mediated by a registered tool.

## Glossary

- **Agent_Gateway**: The unified FastAPI entry point that accepts all incoming agent requests via HTTP REST or WebSocket and routes them to the Agent_Orchestrator
- **Agent_Orchestrator**: The existing core service (`app/services/agent.py`) that manages conversation context, coordinates between the LLM and tools, and drives the reasoning loop; extended in this spec
- **Planner_Agent**: A new sub-agent layer that decomposes complex multi-step user requests into an ordered sequence of subtasks before execution
- **Subtask**: A discrete, independently executable unit of work produced by the Planner_Agent
- **Execution_Plan**: An ordered list of Subtasks produced by the Planner_Agent for a single user request
- **Tool_Router**: A new component that classifies each Tool_Call and dispatches it to the appropriate tool category (service tool, database tool, memory tool, or external tool)
- **Tool_Registry**: The existing centralized tool registry (`app/services/tool_registry.py`)
- **Decision_Engine**: The existing LLM-based tool selection component (`app/services/decision.py`)
- **Tool_Executor**: The existing async tool execution framework (`app/services/tool_executor.py`)
- **User_Service_Tool**: A FastAPI-router-based tool that provides CRUD operations for user entities
- **Task_Service_Tool**: A FastAPI-router-based tool that provides CRUD operations for task entities
- **Database_Tool**: A tool that provides parameterized SQL query execution against the PostgreSQL database via SQLAlchemy
- **Memory_Service_Tool**: A tool that stores and retrieves conversation context chunks using pgvector semantic similarity search
- **Redis_Cache**: The Redis in-memory store used for fast current-state caching and session data
- **Kafka_Broker**: The Apache Kafka event broker used for asynchronous agent event streaming and inter-service communication
- **Kafka_Producer**: The component that publishes agent events to Kafka topics
- **Kafka_Consumer**: The component that subscribes to Kafka topics and processes incoming agent events
- **WebSocket_Manager**: The component that manages active WebSocket connections between the Flutter client and the Agent_Gateway
- **WebSocket_Session**: A persistent bidirectional WebSocket connection associated with a single authenticated user
- **Chat_Session**: A logical unit representing a user's ongoing interaction, tracked in Redis for live state
- **LLM_Provider**: The multi-provider OpenRouter/Groq/NVIDIA/HuggingFace/Google routing layer (existing Fallback_Router)
- **Embedding_Model**: An LLM or dedicated model used to convert text into vector embeddings for semantic search
- **pgvector**: The PostgreSQL extension used for storing and querying high-dimensional vector embeddings
- **Memory_Chunk**: A stored unit of conversation context consisting of text content and its vector embedding
- **Semantic_Search**: Retrieval of Memory_Chunks by cosine similarity between a query embedding and stored embeddings
- **SSE**: Server-Sent Events — the existing streaming transport being superseded by WebSocket for real-time chat
- **FCM**: Firebase Cloud Messaging — push notification delivery to the Flutter app when the user is not connected via WebSocket
- **Correlation_ID**: A UUID that traces a single request across all system components for observability
- **Fallback_Router**: The existing multi-provider LLM routing system with automatic failover
- **Real_Time_Events_Tool**: A read-only tool that lets the agent query current live state (from Redis_Cache) and recent real-time events (buffered from Kafka topics) as a data source during reasoning; corresponds to the "WebSocket Tool / Live Events" node in the reference architecture
- **Live_State**: The current fast-changing value of a keyed entity held in Redis_Cache (e.g., a user's presence, a task's live status), as opposed to permanent data in PostgreSQL
- **Event_Buffer**: A bounded, in-memory ring buffer holding the most recent events consumed from a Kafka topic so the Real_Time_Events_Tool can return them without replaying the full topic
- **Requires_Fresh_Data**: A per-request determination made by the Decision_Engine that a request needs real-time source-of-truth data rather than a cached value; when true, the Agent_Orchestrator bypasses Redis_Cache reads for the entities involved
- **Source_Of_Truth**: The authoritative live origin of a piece of data (a service tool, the Database_Tool, or the Real_Time_Events_Tool), distinct from any cached copy in Redis_Cache
- **Tool_Mediated_Access**: The security invariant that the LLM_Provider never touches databases, credentials, or infrastructure directly; all data and system access occurs exclusively through registered Tool_Calls dispatched by the Tool_Router

## Requirements

### Requirement 1: Agent Gateway Entry Point

**User Story:** As a Flutter client developer, I want a single, stable entry point for all agent interactions, so that the client does not need to know the internal topology of the backend services.

#### Acceptance Criteria

1. THE Agent_Gateway SHALL expose a `/api/agent/chat` HTTP POST endpoint that accepts a Chat_Request and returns a Chat_Response
2. THE Agent_Gateway SHALL expose a `/api/agent/ws/{session_id}` WebSocket endpoint for real-time bidirectional chat
3. WHEN a Chat_Request is received on the HTTP endpoint, THE Agent_Gateway SHALL forward it to the Agent_Orchestrator and return the result within 60 seconds measured from the time the gateway receives the request; IF the total request duration exceeds 60 seconds while actively processing a chat request, THE Agent_Gateway SHALL return HTTP 504 with the message "Request timed out"; THE Agent_Gateway SHALL only return HTTP 504 timeout responses when actively processing a chat request and SHALL NOT return HTTP 504 when no actual chat request is being processed
4. WHEN a WebSocket connection is established, THE Agent_Gateway SHALL register the connection in the WebSocket_Manager with the associated session_id; IF the registration fails due to system issues, THE Agent_Gateway SHALL immediately close the WebSocket connection and SHALL NOT leave the connection open in an unregistered state
5. WHEN a WebSocket connection is closed by the client, THE Agent_Gateway SHALL atomically remove the connection from the WebSocket_Manager and release associated resources such that both operations succeed together or both fail together; partial cleanup SHALL NOT be permitted; THE system MAY also remove connections for reasons other than client closure (such as server-side timeouts, errors, or resource limits) provided that any such removal is also performed atomically
6. THE Agent_Gateway SHALL validate all incoming Chat_Requests against the existing ChatRequest schema before forwarding
7. IF a Chat_Request fails schema validation, THEN THE Agent_Gateway SHALL return an HTTP 422 response with field-level error details including the invalid field names and violation descriptions; THE Agent_Gateway SHALL only return HTTP 422 when schema validation specifically fails and SHALL NOT return HTTP 422 when validation actually passes; all validation failures SHALL result in a non-success status code
8. THE Agent_Gateway SHALL attach a Correlation_ID formatted as a UUID v4 string to every request before forwarding to the Agent_Orchestrator; THE Correlation_ID SHALL be returned in the `X-Correlation-ID` response header
9. WHEN the Agent_Orchestrator returns an error, THE Agent_Gateway SHALL return an HTTP 503 response with `{"error": "Service temporarily unavailable"}` without including stack traces, internal module names, or database error messages; THE Agent_Gateway SHALL only return HTTP 503 when the Agent_Orchestrator specifically returns an error and SHALL NOT return HTTP 503 during normal successful operation

### Requirement 2: WebSocket Real-Time Chat

**User Story:** As a chat user on the Flutter app, I want responses to stream to me in real time over a persistent connection, so that I see tokens appearing as the LLM generates them rather than waiting for the full response.

#### Acceptance Criteria

1. THE WebSocket_Manager SHALL maintain a registry of active WebSocket_Sessions keyed by session_id
2. WHEN the Agent_Orchestrator produces a streaming token, THE WebSocket_Manager SHALL send it to the corresponding WebSocket_Session as a JSON message with `{"type": "token", "content": "..."}` format
3. WHEN a response generation is complete, THE WebSocket_Manager SHALL send a final message with `{"type": "done", "conversation_id": ..., "model": "...", "platform": "..."}` format
4. WHEN a tool execution begins during a streaming response, THE WebSocket_Manager SHALL send a status message with `{"type": "tool_start", "tool_name": "..."}` format
5. WHEN a tool execution completes, THE WebSocket_Manager SHALL send a status message with `{"type": "tool_end", "tool_name": "...", "duration_ms": ...}` format
6. WHEN an error occurs during generation, THE WebSocket_Manager SHALL attempt to send `{"type": "error", "message": "..."}` and SHALL close the WebSocket_Session regardless of whether the error message was successfully delivered
7. THE WebSocket_Manager SHALL support at least 100 concurrent WebSocket_Sessions with a p99 token delivery latency no greater than 200ms measured from the time the token is produced to the time it is sent to the WebSocket send buffer
8. WHEN a WebSocket_Session has received no inbound message for more than 30 minutes, THE WebSocket_Manager SHALL send a `{"type": "ping"}` message to verify the connection is alive
9. IF a ping receives no pong response within 10 seconds, THEN THE WebSocket_Manager SHALL close and remove the WebSocket_Session; IF a pong arrives after the timeout has already been processed, THE WebSocket_Manager SHALL ignore the late pong and keep the session removed
10. WHEN a client sends a Chat_Request via the HTTP `/api/agent/chat` endpoint while also having an active WebSocket_Session for the same session_id, THE Agent_Gateway SHALL stream the response to the WebSocket_Session; WHEN no active WebSocket_Session exists for the session_id AND the system has fewer than 100 concurrent sessions, THE Agent_Gateway SHALL fall back to the existing SSE endpoint behavior; WHEN the system is at or above 100 concurrent sessions, THE Agent_Gateway SHALL reject the request with HTTP 503 regardless of WebSocket availability

### Requirement 3: Planner Agent Layer

**User Story:** As a chat user, I want the system to handle complex multi-step requests like "find all overdue tasks, summarize them, and notify the assignees," so that I can accomplish compound goals with a single message.

#### Acceptance Criteria

1. WHEN a user message is received, THE Planner_Agent SHALL analyze it using the LLM to determine if the request requires multiple distinct steps; a request is multi-step when it explicitly or implicitly requires results from one operation to be used as input to a subsequent distinct operation
2. IF a request is determined to be simple (single step), THEN THE Planner_Agent SHALL pass it directly to the Agent_Orchestrator without creating an Execution_Plan
3. WHEN a multi-step request is detected, THE Planner_Agent SHALL produce an Execution_Plan containing an ordered list of Subtasks
4. EACH Subtask in the Execution_Plan SHALL specify: a natural-language description, a list of required tool names from the Tool_Registry, a list of dependency indices referencing prior Subtask outputs using zero-based integer indices, and a natural-language success criterion
5. THE Agent_Orchestrator SHALL execute Subtasks in ascending index order as defined by the Execution_Plan
6. WHEN a Subtask completes, THE Agent_Orchestrator SHALL make its output available as input to subsequent Subtasks that declare a dependency on it by injecting it into the tool call parameters or prompt context of the dependent Subtask
7. IF a Subtask fails and a subsequent Subtask lists the failed Subtask's index in its dependencies, THEN THE Agent_Orchestrator SHALL mark the dependent Subtask as skipped with reason "dependency {index} failed" and include that reason in the final response
8. WHEN all Subtasks have status completed or skipped, THE Agent_Orchestrator SHALL generate a final response summarizing each Subtask's outcome including its description, status (completed/skipped), and either its output or failure reason
9. THE Planner_Agent SHALL limit Execution_Plans to a maximum of 10 Subtasks per request; IF the LLM generates a plan with more than 10 Subtasks, THE Planner_Agent SHALL truncate to the first 10 and log a warning
10. WHEN an Execution_Plan is created, THE WebSocket_Manager SHALL send a `{"type": "plan_created", "subtask_count": N, "subtasks": [{"index": ..., "description": "..."}]}` message to the active WebSocket_Session for the request's session_id

### Requirement 4: Tool Router

**User Story:** As a developer, I want a centralized dispatching layer that routes tool calls to the correct backend service, so that the Decision_Engine does not need to know the implementation details of each tool category.

#### Acceptance Criteria

1. THE Tool_Router SHALL classify each incoming Tool_Call into one of five categories: service_tool, database_tool, memory_tool, realtime_tool, or external_tool; classification SHALL be based on a static prefix mapping: tool names beginning with `user_` or `task_` map to service_tool, names beginning with `query_` map to database_tool, names beginning with `memory_` map to memory_tool, names beginning with `realtime_` map to realtime_tool, and all other registered names map to external_tool; THE Tool_Router SHALL enforce strict consistency such that the dispatch destination of every Tool_Call matches its prefix-based classification; a tool classified as service_tool SHALL only be dispatched to User_Service_Tool or Task_Service_Tool, and no tool SHALL be dispatched to a handler that does not correspond to its prefix-based category
2. WHEN a Tool_Call is classified as service_tool, THE Tool_Router SHALL dispatch it to either the User_Service_Tool or Task_Service_Tool based on the tool name prefix (`user_` → User_Service_Tool, `task_` → Task_Service_Tool); IF a tool is classified as service_tool, THE Tool_Router SHALL dispatch it to User_Service_Tool or Task_Service_Tool and SHALL NOT dispatch it to any other handler
3. WHEN a Tool_Call is classified as database_tool, THE Tool_Router SHALL dispatch it exclusively to the Database_Tool and SHALL NOT dispatch it to any other handler simultaneously; dispatch to the Database_Tool SHALL be exclusive, meaning the Tool_Router SHALL enforce that each Tool_Call is processed by exactly one handler
4. WHEN a Tool_Call is classified as memory_tool, THE Tool_Router SHALL dispatch it to the Memory_Service_Tool
5. WHEN a Tool_Call is classified as external_tool, THE Tool_Router SHALL dispatch it to the existing Tool_Executor for external API calls (web search, etc.)
6. THE Tool_Router SHALL consult the Tool_Registry to resolve tool names to their registered implementations before dispatching
7. IF a Tool_Call references a tool name not present in the Tool_Registry at dispatch time, THEN THE Tool_Router SHALL return a ToolResult with status "error" and error_message "Tool '{name}' not found in registry"; this error SHALL be returned regardless of whether dispatch was actually attempted; classification SHALL succeed regardless of registry state
8. THE Tool_Router SHALL allow classification of a Tool_Call to succeed even when the tool name is not yet registered in the Tool_Registry, deferring the not-found error to the dispatch step
9. THE Tool_Router SHALL log each routing decision at INFO level including: tool name, resolved category, target handler class name, and the Correlation_ID
10. THE Tool_Router SHALL include the Correlation_ID from the originating request in every downstream ToolCall and ToolResult it produces or forwards
11. WHEN a Tool_Call is classified as realtime_tool, THE Tool_Router SHALL dispatch it to the Real_Time_Events_Tool

### Requirement 5: User Service Tool

**User Story:** As an agent, I want to look up, create, and update user records during a conversation, so that personalized responses can be generated based on user profile data.

#### Acceptance Criteria

1. THE User_Service_Tool SHALL be registered in the Tool_Registry with four operations: `user_get`, `user_list`, `user_create`, and `user_update`
2. WHEN `user_get` is invoked with a user_id, THE User_Service_Tool SHALL return the user record from the database; IF the database response exceeds 2 seconds, THE User_Service_Tool SHALL return the available data with a `degraded: true` field in the ToolResult data rather than failing the request
3. WHEN `user_list` is invoked, THE User_Service_Tool SHALL return a paginated list of user records with a default page size of 20; the caller MAY specify a different page size via a `page_size` parameter, and the response SHALL include `items`, `page`, `page_size`, and `total_count` fields reflecting the actual page size used
4. WHEN `user_create` is invoked with `name`, `email`, and `role` fields, THE User_Service_Tool SHALL create a new user record and return the created entity including its assigned `id`
5. WHEN `user_update` is invoked with a `user_id` and one or more updatable field values, THE User_Service_Tool SHALL apply only the provided fields and return the full updated user record
6. IF `user_get` or `user_update` is invoked with a user_id that does not exist, THEN THE User_Service_Tool SHALL return a ToolResult with status "error" and error_message "User {user_id} not found"
7. IF `user_create` is invoked with an email that already exists in the database, THEN THE User_Service_Tool SHALL return a ToolResult with status "error" and error_message "Email {email} is already registered"
8. THE User_Service_Tool SHALL validate all input parameters against its registered JSON Schema before executing any database operation; IF validation fails, THE User_Service_Tool SHALL return a ToolResult with status "error" and error_message describing the schema violation without executing the database operation; THE User_Service_Tool SHALL only return a successful status when both validation passes and the database operation succeeds; THE User_Service_Tool SHALL return an error status when the database operation fails even if validation passed; all validation failures SHALL result in a non-success status code and SHALL NOT result in any database operation being executed
9. THE User_Service_Tool SHALL never include fields named `password`, `password_hash`, or `api_key` in any ToolResult data payload

### Requirement 6: Task Service Tool

**User Story:** As an agent, I want to query, create, and update tasks during a conversation, so that users can manage their work through natural language.

#### Acceptance Criteria

1. THE Task_Service_Tool SHALL be registered in the Tool_Registry with five operations: `task_get`, `task_list`, `task_create`, `task_update`, and `task_complete`
2. WHEN `task_list` is invoked, THE Task_Service_Tool SHALL return tasks with their `id`, `title`, `description`, `status`, `assignee_id`, `due_date`, and `priority` fields; an optional `status` filter parameter SHALL narrow results to tasks matching the given status value
3. WHEN `task_create` is invoked with `title`, `description`, `assignee_id`, `due_date`, and `priority`, THE Task_Service_Tool SHALL create the task record and return it with its assigned `id`; IF `task_create` is invoked with missing required fields, THE Task_Service_Tool SHALL return a ToolResult with status "error" and error_message identifying the missing fields without executing a database write
4. WHEN `task_complete` is invoked with a `task_id`, THE Task_Service_Tool SHALL set the task `status` to "completed" and record the `completed_at` timestamp as the current UTC time
5. WHEN `task_update` is invoked with a `task_id` and one or more updatable field values, THE Task_Service_Tool SHALL apply only the provided fields and return the full updated task record
6. IF any task operation is invoked with a `task_id` that does not exist, THEN THE Task_Service_Tool SHALL return a ToolResult with status "error" and error_message "Task {task_id} not found"
7. THE Task_Service_Tool SHALL support the following optional filter parameters on `task_list`: `status` (string), `assignee_id` (integer), `due_date_from` and `due_date_to` (ISO 8601 date strings), and `priority` (string)
8. ALL `task_list` responses SHALL include the pagination metadata fields `page_size`, `total_count`, and `next_page_token`; WHEN the total result count exceeds 100, THE Task_Service_Tool SHALL return only the first 100 results and populate `next_page_token` with an opaque cursor for the next page
9. THE Task_Service_Tool SHALL validate all input parameters against its registered JSON Schema before executing database operations; IF validation fails, THE Task_Service_Tool SHALL return a ToolResult with status "error" without executing the database operation; database operations SHALL never be attempted when schema validation has failed

### Requirement 7: Database Tool

**User Story:** As an agent, I want to execute structured database queries during a conversation, so that I can retrieve business data that doesn't fit neatly into a single service operation.

#### Acceptance Criteria

1. THE Database_Tool SHALL be registered in the Tool_Registry with a `query_database` operation that accepts a `query_description` string parameter
2. WHEN `query_database` is invoked, THE Database_Tool SHALL send the `query_description` and the current database schema summary to the LLM and request a parameterized SQL SELECT statement in return
3. WHEN `query_database` is invoked, THE Database_Tool SHALL validate and reject any SQL that contains forbidden DML/DDL keywords before executing; THE Database_Tool SHALL also validate that the generated SQL is syntactically well-formed; IF the SQL is syntactically invalid, THE Database_Tool SHALL return a ToolResult with status "error" and error_message "Generated SQL is syntactically invalid" before performing keyword checking; THE Database_Tool SHALL then validate that the SQL passes an additional validation step beyond syntax and keyword checking before executing; THE Database_Tool SHALL only execute SELECT statements; IF the LLM-generated SQL contains INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, or any DDL keyword, THE Database_Tool SHALL return a ToolResult with status "error" and error_message "Only SELECT queries are permitted" without executing the statement
4. WHEN a valid SELECT query is generated, THE Database_Tool SHALL execute it using the existing SQLAlchemy session and return the results as a list of row dicts; IF the query execution raises a database exception, THE Database_Tool SHALL return a ToolResult with status "error" and a sanitized error_message that omits internal table names and connection details, and SHALL NOT return any partial results; this no-partial-results guarantee SHALL apply to all error conditions including validation failures, SQL keyword violations, and LLM parsing failures — no partial results SHALL be returned for any error condition
5. THE Database_Tool SHALL limit query results to a maximum of 500 rows by capping the `max_rows` parameter at 500; any caller-supplied `max_rows` value above 500 SHALL be silently reduced to 500; WHEN results are truncated, the ToolResult data SHALL include a `truncated: true` field and a `total_available` estimate
6. THE Database_Tool SHALL log the generated SQL query text and execution duration in milliseconds at INFO level for auditability
7. THE Database_Tool SHALL reject any LLM-generated SQL that contains string literals constructed from the original user query text without parameterization; IF such interpolation is detected, THE Database_Tool SHALL return a ToolResult with status "error" and error_message "Parameterization required: direct value interpolation detected" using this exact literal error message string
8. IF the LLM fails to produce a parseable SQL statement from the `query_description`, THE Database_Tool SHALL return a ToolResult with status "error" and error_message "Could not generate a valid SQL query from the provided description"

### Requirement 8: Memory Service Tool and pgvector

**User Story:** As a chat user, I want the agent to remember relevant context from earlier in our conversation and from previous sessions, so that responses are coherent and personalized without me repeating myself.

#### Acceptance Criteria

1. THE Memory_Service_Tool SHALL be registered in the Tool_Registry with operations: `memory_store`, `memory_search`, and `memory_delete`
2. WHEN `memory_store` is invoked with `text` and `conversation_id`, THE Memory_Service_Tool SHALL generate a vector embedding via the Embedding_Model and persist a Memory_Chunk row containing the embedding, `text`, `conversation_id`, `user_id`, and `created_at` timestamp; IF the Embedding_Model call fails, THE Memory_Service_Tool SHALL return a ToolResult with status "error" and error_message "Embedding generation failed" without writing to the database
3. WHEN `memory_search` is invoked with a `query` string and optional `conversation_id` filter, THE Memory_Service_Tool SHALL generate a query embedding and return the top-k Memory_Chunks ranked by cosine similarity; IF embedding generation fails during search, THE Memory_Service_Tool SHALL return a ToolResult with status "error" and error_message "Embedding generation failed"
4. THE Memory_Service_Tool SHALL return a default of 5 top-k results unless `top_k` is specified by the caller, with a hard maximum of 20
5. WHEN `memory_delete` is invoked with a `memory_id`, THE Memory_Service_Tool SHALL remove the corresponding Memory_Chunk row from the pgvector table
6. IF `memory_delete` is invoked with a `memory_id` that does not exist, THEN THE Memory_Service_Tool SHALL return a ToolResult with status "error" and error_message "Memory {memory_id} not found"; IF `memory_delete` is invoked with a `memory_id` that exists but belongs to a different user or conversation, THE Memory_Service_Tool SHALL return a ToolResult with status "error" and error_message "Access denied" without deleting the record
7. THE Agent_Orchestrator SHALL invoke `memory_store` at the end of each conversation turn to persist the user message and assistant response; IF this automatic store operation fails, THE Agent_Orchestrator SHALL log the failure at WARNING level and continue without interrupting the conversation
8. THE Agent_Orchestrator SHALL invoke `memory_search` at the start of each conversation turn and prepend any returned Memory_Chunk texts to the system prompt as context under a `## Relevant Memory` heading; memory context SHALL only be prepended when a conversation turn starts and the automatic search is invoked, and SHALL NOT be prepended in any other situation
9. THE Memory_Service_Tool SHALL exclude `memory_search` results with a cosine similarity score below the configured threshold (default 0.7); the threshold SHALL be configurable via the `MEMORY_SIMILARITY_THRESHOLD` environment variable with valid range [0.0, 1.0]
10. FOR ALL valid text inputs where the Memory_Chunk was previously stored successfully, storing a Memory_Chunk and then immediately searching with the same text as the query SHALL return that chunk with a similarity score ≥ the configured threshold; this guarantee applies only to text that was successfully stored and does not apply to arbitrary semantic search operations on text that was never stored
11. THE Memory_Service_Tool SHALL support a `memory_store_batch` operation that accepts a list of up to 50 `{text, conversation_id}` objects and persists them in a single database transaction
12. WHEN `memory_search` is invoked with a `top_k` value greater than the total number of stored Memory_Chunks that meet the similarity threshold, THE Memory_Service_Tool SHALL return all qualifying chunks without error

### Requirement 9: Redis Cache Integration

**User Story:** As a system operator, I want live agent session state and frequently-accessed data cached in Redis, so that the agent can respond to repeated or related queries without redundant database round-trips.

#### Acceptance Criteria

1. THE Redis_Cache SHALL store active Chat_Session state as a JSON-serialized object keyed by `session:{session_id}` with a TTL of 1800 seconds (30 minutes)
2. WHEN the Agent_Orchestrator starts processing a request, THE Redis_Cache SHALL be queried for `session:{session_id}` before issuing any database query for conversation history
3. WHEN the Agent_Orchestrator completes a response, THE Redis_Cache SHALL write the updated Chat_Session and reset its TTL to 1800 seconds
4. THE Redis_Cache SHALL cache the result of each `user_get` operation keyed by `user:{user_id}` with a TTL of 300 seconds (5 minutes)
5. WHEN a `user_update` operation completes successfully, THE Redis_Cache SHALL delete the `user:{user_id}` key to invalidate the cached record
6. THE Redis_Cache SHALL cache the serialized enabled-tools list from the Tool_Registry keyed by `tools:enabled` with a TTL of 60 seconds
7. IF the Redis connection is unavailable when the Agent_Orchestrator attempts a cache read or write, THEN THE Agent_Orchestrator SHALL fall back to the equivalent direct database query, continue processing normally, and log the unavailability at WARNING level without returning an error to the user; Redis unavailability MAY appear in admin interfaces or detailed error responses but SHALL NOT surface in standard user-facing responses or break user workflows
8. THE Redis_Cache SHALL be configured with a connection pool of minimum 5 and maximum 20 connections
9. WHEN any individual Redis operation does not complete within 500ms, THE Redis_Cache SHALL treat it as a cache miss, abort the operation, and proceed without the cached value; WHEN the cache times out or encounters operational issues, THE system SHALL fall back to an equivalent database query
10. WHEN a Redis_Cache write fails after the Agent_Orchestrator has already generated and returned a response, THE Agent_Orchestrator SHALL log the cache write failure at WARNING level and SHALL NOT retry or surface the failure to the user
11. WHEN any Redis operational error other than connection unavailability occurs (e.g., authentication failure, pool exhaustion, protocol error), THE Redis_Cache SHALL log the error at WARNING level and treat the affected operation as a cache miss

### Requirement 10: Kafka Event Streaming

**User Story:** As a system operator, I want all significant agent events published to a Kafka topic, so that other services can react to agent activity in real time and I can build audit trails and analytics.

#### Acceptance Criteria

1. THE Kafka_Producer SHALL publish an event to the `agent.events` topic at each of the following lifecycle moments: `request_received`, `plan_created`, `tool_started`, `tool_completed`, `tool_failed`, `response_generated`
2. EACH Kafka event message SHALL be a JSON object containing at minimum: `event_type` (string), `correlation_id` (UUID v4 string), `conversation_id` (integer or null), `session_id` (string), `timestamp_utc` (ISO 8601 string), and `payload` (object with event-specific fields)
3. WHEN a `tool_completed` event is published, its `payload` SHALL include: `tool_name` (string), `duration_ms` (number), and `status` ("success" | "error" | "timeout")
4. THE Kafka_Producer SHALL use acks=all (at-least-once delivery) and SHALL not consider an event published until it receives broker acknowledgment
5. IF the Kafka broker is unreachable or acknowledgment is not received within 5 seconds, THEN THE Kafka_Producer SHALL log the event loss at WARNING level with the correlation_id and continue processing the agent request without blocking
6. THE Kafka_Consumer SHALL subscribe to the `agent.commands` Kafka topic on startup and process inbound command messages in a background task
7. WHEN a message with `command_type: "notify_user"` is consumed from `agent.commands`, THE Kafka_Consumer SHALL extract `session_id` and `payload` and call `WebSocket_Manager.send(session_id, payload)` to deliver the notification
8. THE Kafka_Producer SHALL use a linger time of 100ms and a batch size of up to 16KB to optimize throughput without exceeding the linger budget
9. THE Kafka bootstrap server address SHALL be read from the `KAFKA_BOOTSTRAP_SERVERS` environment variable; the Kafka client SHALL not start and SHALL log a startup error if the variable is absent and the default `localhost:9092` is unreachable

### Requirement 11: FCM Push Notifications

**User Story:** As a Flutter app user, I want to receive push notifications for important agent events when the app is in the background, so that I am informed of task completions and alerts without keeping the app open.

#### Acceptance Criteria

1. THE system SHALL integrate with Firebase Cloud Messaging to deliver push notifications to the Flutter app
2. WHEN a `task_completed` agent event is generated and the target user has no active WebSocket_Session, THE system SHALL send an FCM push notification to the user's registered device token; WHEN a `task_completed` event is generated and the target user has no registered device token, THE system SHALL skip sending the FCM notification without error
3. WHEN a `plan_created` event involves a multi-step plan of 3 or more Subtasks (i.e., Subtask count ≥ 3), THE system SHALL send an FCM push notification indicating that the agent is working on a complex request; WHEN a plan has fewer than 3 Subtasks (including zero Subtasks), THE system SHALL NOT send an FCM push notification for plan creation; zero-subtask plans SHALL be treated the same as other plans with fewer than 3 Subtasks and SHALL NOT trigger a notification
4. THE system SHALL store FCM device tokens in the database associated with the user record
5. IF an FCM delivery attempt fails, THEN THE system SHALL retry up to 3 times with exponential backoff; WHEN all retries are exhausted, THE system SHALL automatically set the delivery status to FAILED and log a permanent failure
6. THE system SHALL not send FCM notifications when the user has an active WebSocket_Session, to avoid duplicate notifications
7. WHEN a user logs out, THE system SHALL remove only the FCM device token associated with the logging-out device, leaving tokens for any other active devices unchanged

### Requirement 12: Planner Agent Planning Quality

**User Story:** As a chat user, I want the Planner_Agent to produce coherent, efficient plans that do not repeat or contradict themselves, so that complex requests are completed correctly on the first attempt.

#### Acceptance Criteria

1. THE Planner_Agent SHALL include the full list of available tools in the planning prompt so that Subtasks only reference tools that exist in the Tool_Registry
2. WHEN an Execution_Plan is generated, THE Planner_Agent SHALL verify that no two Subtasks reference conflicting write operations on the same resource
3. THE Planner_Agent SHALL assign each Subtask a unique integer index starting at 1
4. WHEN a Subtask declares input dependencies, THE Planner_Agent SHALL verify that all referenced dependency indices are lower than the Subtask's own index
5. THE Planner_Agent SHALL re-plan up to `max_retries` times (default 2) if the generated Execution_Plan fails its own validation checks before returning an error; the retry limit SHALL be honored regardless of the configured `max_retries` value
6. WHEN planning fails after exhausting all retries, THE Planner_Agent SHALL fall back to single-step execution and log a warning; THE Planner_Agent SHALL trigger the single-step fallback and log a warning whenever validation fails, regardless of planning success status; THE Planner_Agent SHALL NOT log warnings or use single-step mode when planning succeeds and validation passes
7. FOR ALL valid multi-step requests, the sum of Subtask indices in any valid Execution_Plan SHALL equal the sum of integers from 1 to N where N is the number of Subtasks (index uniqueness invariant)

### Requirement 13: Memory Round-Trip and Search Quality

**User Story:** As a developer, I want the memory system to reliably store and retrieve context, so that the agent's recall is consistent and testable.

#### Acceptance Criteria

1. THE Memory_Service_Tool SHALL use cosine similarity as the distance metric for all Semantic_Search operations
2. WHEN a Memory_Chunk is stored and then searched using its original text as the query, THE Memory_Service_Tool SHALL return that chunk with a similarity score above the configured threshold; this guarantee applies only to chunks that were successfully stored and subsequently searched, not to arbitrary semantic search operations
3. THE Memory_Service_Tool SHALL store the conversation_id, user_id, creation timestamp, and raw text alongside each vector embedding
4. WHEN two Memory_Chunks with identical text content are stored, THE Memory_Service_Tool SHALL store them as separate entries with distinct memory_ids and SHALL create separate database records for each; IF separate database records are not created for identical text inputs, the requirement SHALL be considered violated regardless of whether distinct memory_ids were assigned; the requirement is only met when both distinct IDs and separate database entries exist
5. THE Memory_Service_Tool SHALL support batch `store_memory` calls that accept a list of up to 50 text items and store them in a single database transaction
6. WHEN `search_memory` is called with a top_k value greater than the total number of stored Memory_Chunks, THE Memory_Service_Tool SHALL return all stored chunks without error

### Requirement 14: Agent Gateway Security

**User Story:** As a system operator, I want the Agent_Gateway to enforce authentication and rate limiting, so that unauthorized access is prevented and costs are controlled.

#### Acceptance Criteria

1. THE Agent_Gateway SHALL require a valid API key or session token on all `/api/agent/*` endpoints
2. WHEN an unauthenticated request is received, THE Agent_Gateway SHALL return HTTP 401 with the message "Authentication required" and SHALL immediately stop all further processing of the request beyond authentication checking
3. THE Agent_Gateway SHALL enforce a rate limit of 60 requests per minute per authenticated user on the HTTP endpoint; WHEN determining whether a user has exceeded their limit, THE Agent_Gateway SHALL consider recent request history even when the user's current request count appears to be zero
4. WHEN a user exceeds the rate limit, THE Agent_Gateway SHALL return HTTP 429 with a `Retry-After` header indicating the seconds until the next request is allowed
5. THE Agent_Gateway SHALL validate WebSocket upgrade requests for valid session tokens before completing the handshake; THE Agent_Gateway SHALL only accept tokens that are session tokens and SHALL reject any valid token that is not a session token
6. IF a WebSocket upgrade request carries an invalid, expired, missing, or non-session token, THEN THE Agent_Gateway SHALL reject the upgrade with HTTP 401; THE Agent_Gateway SHALL treat missing tokens the same as invalid tokens and SHALL reject upgrades when no token is provided at all
7. THE Agent_Gateway SHALL log all authentication failures including the source IP address and the endpoint attempted

### Requirement 15: Configuration and Environment

**User Story:** As a developer, I want all infrastructure connection strings and feature flags configurable via environment variables, so that the system can be deployed across development, staging, and production without code changes.

#### Acceptance Criteria

1. THE system SHALL load the Redis connection URL from the `REDIS_URL` environment variable with a default of `redis://localhost:6379`
2. THE system SHALL load the Kafka bootstrap servers from the `KAFKA_BOOTSTRAP_SERVERS` environment variable with a default of `localhost:9092`
3. THE system SHALL load the pgvector database connection string from the existing `DATABASE_URL` environment variable
4. THE system SHALL load the FCM service account credentials path from the `FCM_CREDENTIALS_PATH` environment variable
5. THE system SHALL load the Embedding_Model identifier from the `EMBEDDING_MODEL` environment variable with a default of `text-embedding-3-small`
6. WHEN a required environment variable is missing at startup, THE system SHALL always log a descriptive warning regardless of whether the default value works, then use the documented default value; IF the service pointed to by the default value is unreachable at startup, THE system SHALL immediately fail startup with a descriptive error treating it as a critical configuration error, even when the environment variable was intentionally absent; this failure behavior applies regardless of whether the user provides an explicit connection string or relies on defaults
7. THE system SHALL support a `AGENT_FEATURES` environment variable that accepts a comma-separated list of feature flags: `planner`, `redis_cache`, `kafka`, `fcm`, `websocket`
8. WHEN a feature flag is absent from `AGENT_FEATURES`, THE system SHALL disable that feature and fall back to the equivalent existing behavior (e.g., direct DB queries instead of Redis cache)

### Requirement 16: Observability and Health

**User Story:** As a system operator, I want health check endpoints and structured logs for each new component, so that I can monitor system health in production dashboards.

#### Acceptance Criteria

1. THE system SHALL expose a `/api/agent/health` GET endpoint that returns the health status of: Agent_Gateway, Redis_Cache, Kafka_Producer, Memory_Service_Tool (pgvector), and WebSocket_Manager
2. WHEN all components explicitly report healthy status, THE `/api/agent/health` endpoint SHALL return HTTP 200 with `{"status": "healthy", "components": {...}}`; THE endpoint SHALL NOT return a healthy status unless all components have explicitly reported healthy — if any component is in an unknown, initializing, degraded, or any non-healthy state, or if any system-level inconsistency is detected between individual component health and the overall system state, the overall status SHALL be "degraded"; the "healthy" and "degraded" states are mutually exclusive
3. WHEN one or more components are degraded, THE `/api/agent/health` endpoint SHALL return HTTP 200 with `{"status": "degraded", "components": {...}}` and include the failure reason per component
4. THE system SHALL emit structured JSON logs for every agent request including: correlation_id, session_id, conversation_id, planner_used, subtask_count, tool_calls_made, total_duration_ms, and llm_provider
5. THE system SHALL maintain counters for: WebSocket_Sessions active, Kafka events published, Redis cache hits, Redis cache misses, memory chunks stored, and memory searches performed
6. THE system SHALL expose these counters at a `/api/agent/metrics` GET endpoint in Prometheus text format
7. WHEN a component health check takes longer than 2 seconds to respond, THE `/api/agent/health` endpoint SHALL mark that component as both "timeout" and "degraded" and set the overall health status to "degraded", and SHALL continue returning results for other components; IF a component is already degraded for other reasons and also times out, THEN the endpoint SHALL report both the degradation reason and the timeout condition for that component; WHEN a component health check times out, the overall system status SHALL be set to "degraded" regardless of the component's self-reported status

### Requirement 17: Backward Compatibility

**User Story:** As a Flutter client developer, I want the existing `/api/chat/send` and `/api/chat/stream` endpoints to continue working unchanged, so that I can migrate to the new Agent_Gateway incrementally.

#### Acceptance Criteria

1. THE existing `/api/chat/send` POST endpoint SHALL continue to accept ChatRequest and return ChatResponse without schema changes
2. THE existing `/api/chat/stream` SSE endpoint SHALL continue to stream responses in the existing JSON chunk format
3. WHEN the new agent features are enabled via `AGENT_FEATURES`, THE existing `/api/chat/send` and `/api/chat/stream` endpoints SHALL route requests through the new agent pipeline including the Planner_Agent, Tool_Router, Redis_Cache, and Kafka_Producer; WHEN agent features are disabled via `AGENT_FEATURES`, THE system SHALL immediately route all new requests through the existing Agent_Orchestrator, guaranteeing that at least one active pipeline is always handling requests; IF the new agent pipeline fails to initialize while agent features are enabled, THE system SHALL keep the existing Agent_Orchestrator running until the new pipeline successfully starts — the system SHALL continue retrying pipeline initialization indefinitely while agent features remain enabled, without giving up or automatically disabling agent features; THE system SHALL NOT route any requests to the new pipeline until its initialization is complete; IF the new agent pipeline is active when the `AGENT_FEATURES` flag is disabled, all new requests SHALL switch to the existing Agent_Orchestrator immediately while in-progress requests through the new pipeline SHALL be allowed to complete; WHEN agent features are disabled and the existing Agent_Orchestrator fails to start, THE system SHALL NOT attempt to activate the new pipeline as a fallback and SHALL require the existing Agent_Orchestrator to be available
4. THE new Database_Tool, User_Service_Tool, Task_Service_Tool, Memory_Service_Tool, and Real_Time_Events_Tool SHALL be registered in the existing Tool_Registry and discoverable by the existing Decision_Engine
5. THE existing SQLAlchemy models and migration history SHALL not be modified by this feature; new tables SHALL be added via additive Alembic migrations only

### Requirement 18: Real-Time Events and Live State Tool

**User Story:** As an agent, I want to fetch current live state and recent real-time events during a conversation, so that I can answer questions about what is happening right now (e.g., "is this task still in progress?", "what changed in the last few minutes?") using fresh data rather than permanent-store snapshots.

#### Acceptance Criteria

1. THE Real_Time_Events_Tool SHALL be registered in the Tool_Registry with two read-only operations: `realtime_get_state` and `realtime_recent_events`
2. WHEN `realtime_get_state` is invoked with a `key`, THE Real_Time_Events_Tool SHALL return the current Live_State value held in the Redis_Cache for that key together with an `as_of` UTC timestamp; IF no value exists for the key, THE Real_Time_Events_Tool SHALL return a ToolResult with status "error" and error_message "No live state for '{key}'"
3. WHEN `realtime_recent_events` is invoked with a `topic` and an optional `limit` (default 20, hard maximum 100), THE Real_Time_Events_Tool SHALL return the most recent events for that topic from the Event_Buffer ordered newest-first, each event including `event_type`, `timestamp_utc`, `correlation_id`, and `payload`
4. THE Real_Time_Events_Tool SHALL consume events from subscribed Kafka topics into a bounded Event_Buffer that retains at most the last N events per topic, where N is configurable via the `REALTIME_EVENT_BUFFER_SIZE` environment variable with a default of 500; WHEN the buffer is full, THE Real_Time_Events_Tool SHALL discard the oldest event and log a metric or warning indicating the event was discarded due to buffer overflow
5. THE Real_Time_Events_Tool SHALL be read-only and SHALL NOT publish, modify, or delete events or state; IF a write-style operation is requested, THE Real_Time_Events_Tool SHALL return a ToolResult with status "error" and error_message "Real-time events tool is read-only"
6. IF the Redis_Cache is unavailable when `realtime_get_state` is invoked, THEN THE Real_Time_Events_Tool SHALL attempt an equivalent Source_Of_Truth lookup via the appropriate service tool; if no source is available, THE Real_Time_Events_Tool SHALL return a ToolResult with status "error" and error_message "Live state source unavailable", and the Agent_Orchestrator SHALL log the unavailability at WARNING level and continue
7. IF the Kafka event source is unavailable or no events have been buffered when `realtime_recent_events` is invoked, THEN THE Real_Time_Events_Tool SHALL return a ToolResult with status "error" and error_message "Real-time event source unavailable"; the Agent_Orchestrator SHALL treat this as a degraded, non-fatal condition and continue generating a response
8. WHEN the `kafka` or `redis_cache` feature flag is absent from `AGENT_FEATURES`, THE Real_Time_Events_Tool SHALL report itself as unavailable in `/api/agent/health` and SHALL return a ToolResult with status "error" and error_message "Real-time features are disabled" for all operations rather than failing the overall request; the ToolResult SHALL also set a formal ERROR status in the tool result status field
9. THE Real_Time_Events_Tool SHALL log each invocation at INFO level including the operation, key or topic, result count, and the Correlation_ID; logging SHALL occur only when operations are actually invoked and SHALL NOT be emitted as a constant background behavior

### Requirement 19: Tool-Mediated Data Access and Least Privilege

**User Story:** As a security engineer, I want every data and system interaction to be mediated by a registered tool, so that the LLM cannot reach databases, credentials, or infrastructure directly and the blast radius of a prompt-injection or hallucinated instruction is bounded.

#### Acceptance Criteria

1. THE LLM_Provider SHALL NOT be given direct database connections, raw credentials, or infrastructure handles; all data and system access SHALL occur exclusively through Tool_Calls dispatched by the Tool_Router (Tool_Mediated_Access invariant)
2. THE system SHALL NOT execute any LLM-produced text as shell commands, arbitrary code, or file-system operations; the only executable output accepted from the LLM SHALL be structured Tool_Calls that reference tools registered in the Tool_Registry, plus the constrained read-only SELECT path of the Database_Tool defined in Requirement 7
3. EVERY tool that reads or writes the database SHALL use the existing parameterized SQLAlchemy session; NO tool SHALL construct a query from unparameterized LLM-generated or user-supplied text, consistent with Requirement 7.7
4. Secrets and credentials — including API keys, database passwords, FCM service-account files, and provider tokens — SHALL be injected from environment variables or a secret store at the tool boundary and SHALL NEVER appear in prompts, Tool_Calls, Tool_Results, structured logs, WebSocket messages, or Kafka event payloads
5. WHEN the Tool_Router resolves a Tool_Call to a tool that is registered but disabled in the Tool_Registry, THE Tool_Router SHALL return a ToolResult with status "error" and error_message "Tool '{name}' is disabled" without invoking any handler
6. EACH service tool and the Database_Tool SHALL enforce authorization scoped to the requesting user; IF a Tool_Call attempts to read or modify a resource outside the requesting user's permitted scope, THE tool SHALL return a ToolResult with status "error" and error_message "Access denied" and SHALL NOT return the resource data
7. THE system SHALL write an audit log entry for every privileged write operation (any `*_create`, `*_update`, `*_complete`, or `*_delete` tool invocation) including the Correlation_ID, tool name, acting user identifier, target resource identifier, and outcome
8. THE Agent_Orchestrator SHALL treat all tool inputs and tool results as untrusted data rather than instructions, and SHALL NOT allow content returned inside a Tool_Result to alter the set of tools the LLM is permitted to call or to escalate authorization scope

### Requirement 20: Live Data Freshness

**User Story:** As a chat user, I want time-sensitive questions answered with live source-of-truth data, so that real-time queries are never answered from stale cache while routine lookups still benefit from caching.

#### Acceptance Criteria

1. WHEN the Decision_Engine sets Requires_Fresh_Data for a request, THE Agent_Orchestrator SHALL bypass Redis_Cache reads for the entities involved and fetch directly from the Source_Of_Truth service tool, Database_Tool, or Real_Time_Events_Tool; a fresh data request is only satisfied when the cache bypass succeeds AND data is successfully fetched from the designated Source_Of_Truth tool
2. WHEN data is fetched live, THE resulting ToolResult data SHALL include a `fetched_at` UTC timestamp and a `source: "live"` field; WHEN a value is served from Redis_Cache, THE ToolResult data SHALL include `source: "cache"` and an `age_seconds` field indicating how long ago the cached value was written
3. WHEN a request is marked Requires_Fresh_Data, THE `user:{user_id}` cache read defined in Requirement 9.4 SHALL be bypassed and the freshly fetched value SHALL repopulate the cache with a reset TTL
4. THE results of the Real_Time_Events_Tool and the Database_Tool SHALL always be treated as live and SHALL NOT be served from Redis_Cache; WHEN these sources are unavailable and the Agent_Orchestrator falls back to cached data, the ToolResult SHALL mark the data as `source: "cache"` and `degraded: true` to indicate actual freshness rather than treating it as live; IF data is successfully fetched live from these tools but a subsequent processing error occurs, THE system SHALL fail the request — the entire request SHALL be considered failed when a processing error occurs after live data is fetched, and the live data SHALL NOT be returned in a success response
5. IF live data cannot be fetched because the Source_Of_Truth tool itself is unavailable (not merely a general fetch failure) and only a cached value exists, THEN THE Agent_Orchestrator SHALL return the cached value marked `source: "cache"`, `degraded: true`, and its `age_seconds`, and SHALL note the potential staleness in the response rather than failing the request, consistent with Requirement 5.2; cache-related constraints such as source type and age validation SHALL only apply during this failure condition when live data cannot be fetched; WHEN the Source_Of_Truth is available, cache constraint enforcement is not required
6. THE Requires_Fresh_Data determination SHALL be recorded as a `requires_fresh_data` boolean field in the structured request log defined in Requirement 16.4
