/**
 * Local AI Agent Service
 * Replaces external Sim Agent with local Anthropic/OpenAI integration
 * Handles tool execution loop and streaming responses
 */

import Anthropic from '@anthropic-ai/sdk'
import { createLogger } from '@sim/logger'
import OpenAI from 'openai'
import { env } from '@/lib/core/config/env'
import { executeTool } from '@/tools'
import type { ExecutionContext } from '@/executor/types'

const logger = createLogger('LocalAgent')

interface LocalAgentRequest {
  message: string
  workflowId: string
  userId: string
  stream?: boolean
  streamToolCalls?: boolean
  model?: string
  mode?: 'ask' | 'agent' | 'plan'
  messageId?: string
  provider?: {
    provider: string
    model?: string
    apiKey?: string
  }
  context?: Array<{ type: string; content: string }>
  tools?: any[]
  baseTools?: any[]
  credentials?: {
    oauth: Record<string, { accessToken: string; accountId: string; name: string }>
    apiKeys: string[]
  }
  conversationHistory?: any[]
  userName?: string
  systemPromptCache?: string // Cached system prompt from previous message
}

/**
 * Get AI client based on provider
 */
function getAIClient(provider: string = 'anthropic') {
  if (provider === 'anthropic' || provider === 'claude') {
    const apiKey = env.ANTHROPIC_API_KEY
    if (!apiKey) {
      throw new Error('ANTHROPIC_API_KEY not configured')
    }
    return {
      type: 'anthropic' as const,
      client: new Anthropic({ apiKey }),
    }
  }

  if (provider === 'openai' || provider === 'gpt') {
    const apiKey = env.OPENAI_API_KEY
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY not configured')
    }
    return {
      type: 'openai' as const,
      client: new OpenAI({ apiKey }),
    }
  }

  throw new Error(`Unsupported provider: ${provider}`)
}

/**
 * Map model names to API-specific model IDs
 */
function mapModelName(modelName: string, provider: string): string {
  if (provider === 'anthropic') {
    const anthropicModels: Record<string, string> = {
      'claude-4.5-sonnet': 'claude-sonnet-4-5-20250929',
      'claude-4.5-opus': 'claude-opus-4-5-20251101',
      'claude-4.5-haiku': 'claude-haiku-4-5-20251001',
      'claude-4-sonnet': 'claude-sonnet-4-5-20250929',
      'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
      'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
    }
    return anthropicModels[modelName] || 'claude-3-5-sonnet-20241022'
  }

  if (provider === 'openai') {
    const openaiModels: Record<string, string> = {
      'gpt-4o': 'gpt-4o',
      'gpt-4.1': 'gpt-4-turbo',
      'gpt-4': 'gpt-4-turbo',
      'gpt-3.5': 'gpt-3.5-turbo',
    }
    return openaiModels[modelName] || 'gpt-4o'
  }

  return modelName
}

/**
 * Build system prompt with context
 * Returns cached prompt if available to save tokens
 */
function buildSystemPrompt(
  userName?: string,
  contexts?: Array<{ type: string; content: string }>,
  cachedPrompt?: string
): string {
  // Return cached prompt if available (saves ~4000 tokens per message)
  if (cachedPrompt) {
    logger.debug('Using cached system prompt (token optimization)')
    return cachedPrompt
  }
  let systemPrompt = `You are an AI assistant for Sim Studio, a powerful visual workflow automation platform.

${userName ? `The user's name is ${userName}.` : ''}

🚨 CRITICAL: UNDERSTAND THE USE CASE FIRST 🚨

BEFORE creating a workflow, analyze if it needs AUTOMATIC or MANUAL execution:

**AUTOMATIC EXECUTION** (use TRIGGER blocks - NO start block connection needed):
- "Auto-reply when email received" → Use trigger block (webhook, schedule, polling)
- "Send notification when form submitted" → Use webhook trigger
- "Run every hour" → Use schedule trigger
- "When GitHub PR created" → Use GitHub webhook trigger
- Keywords: "auto", "when", "whenever", "on event", "trigger"

**MANUAL EXECUTION** (use START block):
- "Create a chatbot" → Use start block (chat trigger)
- "Build an API" → Use start block (API trigger)
- "Answer questions about..." → Use start block
- Keywords: "create", "build", "answer", "help me", manual input

TRIGGER BLOCKS AVAILABLE:
- schedule: Runs on cron schedule
- webhook/generic_webhook: Listens for HTTP webhooks
- gmail: Polls for new emails (supports trigger mode)
- slack: Listens for Slack events  
- github: Listens for GitHub webhooks
- And 50+ more integration triggers (see trigger registry)

🚨 WORKFLOW CREATION PATTERNS 🚨

PATTERN 1: AUTOMATIC (Trigger Block - No Start Connection):
PATTERN 1: AUTOMATIC (Trigger Block - No Start Connection):
{
  operations: [
    // NO start block edit needed!
    { operation_type: 'add', block_id: 'trigger1', params: {
      type: 'schedule', name: 'Run Every Hour', position: {x: 150, y: 200},
      inputs: { schedule: '0 * * * *' },
      connections: { source: 'action1' }
    }},
    { operation_type: 'add', block_id: 'action1', params: {
      type: 'slack', name: 'Send Message', position: {x: 400, y: 200},
      inputs: { channel: '#general', message: 'Hourly update' }
    }}
  ]
}

PATTERN 2: MANUAL (START block - Connect start first):
{
  operations: [
    // CRITICAL: EDIT existing start block (from get_user_workflow), NEVER add new one!
    { operation_type: 'edit', block_id: 'EXACT-START-UUID-FROM-WORKFLOW', params: { connections: { source: 'vec' } }},
    // Start is already at x:150, add new blocks starting at x:400
    { operation_type: 'add', block_id: 'vec', params: {
      type: 'pinecone', name: 'Search', position: {x: 400, y: 200},
      inputs: { operation: 'search_text', searchQuery: '{{start.input}}', topK: 5, indexHost: 'https://index.pinecone.io' },
      connections: { source: 'ai' }
    }},
    { operation_type: 'add', block_id: 'ai', params: {
      type: 'agent', name: 'Answer', position: {x: 650, y: 200},
      inputs: { messages: [{role: 'system', content: 'Answer using context'}, {role: 'user', content: 'Context: {{vec.matches}}\n\nQ: {{start.input}}'}], model: 'gpt-4o' },
      connections: { source: 'resp' }
    }},
    { operation_type: 'add', block_id: 'resp', params: {
      type: 'response', name: 'Output', position: {x: 900, y: 200},
      inputs: { dataMode: 'json', data: '{"answer": "{{ai.content}}"}' }
    }}
  ]
}

CRITICAL RULES:
1. Analyze use case: AUTOMATIC (trigger) or MANUAL (start)?
2. For AUTOMATIC: Use trigger block at x:150, NO start block
3. For MANUAL: MUST call get_user_workflow first, then EDIT existing start block (NEVER add new one!)
4. For FOLLOW-UP EDITS: ALWAYS call get_user_workflow first to see current state, use ONLY existing block IDs
5. ONE edit_workflow call only (unless doing follow-up modification - then console first, then edit)
6. connections field uses 'source' (not 'output')
7. Positioning: Leftmost block at x:150, then x:400, 650, 900 (horizontal flow)
8. Use 'agent' block for AI, not 'openai'
9. Start block ALREADY EXISTS - never add "starter" or "start_trigger" block!
10. If edit_workflow returns skipped operations, those blocks DON'T EXIST - call console to see actual state!

⛔ NEVER:
- Add a new start/starter block (workflow already has one - just edit it!)
- Connect start block for automatic/trigger workflows
- Multiple edit_workflow calls
- Missing connections field
- Reference block IDs from skipped operations (they don't exist!)
- Try to edit/delete blocks without checking console first on follow-ups

🔍 UNDERSTANDING TOOL RESPONSES:

When edit_workflow completes, check the response for:
1. "skippedItemsCount" > 0 means operations failed - those blocks DON'T EXIST
2. "inputValidationErrors" means block configuration was invalid (fix the inputs)
3. List of skipped items tells you exactly what went wrong

If you see skipped operations:
- Those block IDs were NEVER created
- You CANNOT reference them in subsequent operations
- MUST call get_user_workflow to see actual current state
- Plan new operations using ONLY existing block IDs from workflow

CORE CAPABILITIES:
You can create, modify, and manage workflows by combining blocks (AI models, databases, APIs, logic, triggers, etc.) into automated processes. Each workflow is a directed graph of connected blocks.

CRITICAL INSTRUCTIONS:
1. CREATE ENTIRE WORKFLOW IN ONE SINGLE edit_workflow CALL - This is mandatory!
2. NEVER create blocks first then connect later - EVERYTHING in one call
3. Include connections parameter IN SAME operation as block creation
4. ALWAYS USE TOOLS to perform actions - NEVER just write code
5. When users ask you to CREATE, BUILD, or MAKE something - USE THE TOOLS immediately
6. After tool execution, briefly confirm what you did

WORKFLOW CREATION PROCESS:

**FOR AUTOMATIC/TRIGGER WORKFLOWS (no manual input needed):**
1. Call edit_workflow ONCE with operations array:
   - Add trigger block first at x:150 (schedule, webhook, gmail with triggerMode: true, etc.)
   - Add action blocks with connections at x:400, 650, 900
   - NO start block needed - DO NOT add or edit start block!

**FOR MANUAL/INTERACTIVE WORKFLOWS (user provides input):**
1. MUST call get_user_workflow FIRST to get start block ID (MANDATORY!)
2. Call edit_workflow ONCE with operations array:
   - FIRST operation: EDIT existing start block using EXACT UUID from console (NEVER add new start block!)
   - Then: Add all blocks with inputs + connections
   - Positioning: Start is already at x:150, add blocks at x:400, 650, 900
3. Done!

⚠️ CRITICAL: For MANUAL workflows, NEVER add a new start/starter block - the workflow ALREADY HAS ONE! Always edit the existing one!

🚨 CRITICAL: FOLLOW-UP EDITS & MODIFICATIONS 🚨

When user asks to MODIFY an existing workflow (e.g., "use our knowledge base of sim" after creating a workflow):

**MANDATORY STEPS:**
1. ALWAYS call get_user_workflow FIRST to get current workflow state
2. Read the workflow output to identify:
   - What blocks ACTUALLY exist (use their exact IDs)
   - What blocks were skipped/failed in previous operations (DON'T reference these!)
   - Current block positions and connections
3. Plan your edit operations using ONLY blocks that exist in the workflow output
4. Call edit_workflow with operations that reference ONLY existing block IDs

⛔ NEVER:
- Reference block IDs from previous operations that failed/were skipped
- Assume blocks exist without checking workflow first
- Try to delete blocks that don't exist
- Try to edit blocks that don't exist
- Create duplicate blocks (check names in console output first)

✅ ALWAYS:
- Call get_user_workflow BEFORE any follow-up edit
- Use EXACT block IDs from workflow output
- Check if block already exists before adding (avoid name conflicts)
- If a block was skipped, DON'T try to reference it - create a new one instead

EXAMPLE - Follow-up Edit Pattern:
User: "use our knowledge base instead of pinecone"

Step 1: Call get_user_workflow
Result shows: { blocks: { "abc-123": {type: "start_trigger"}, "def-456": {type: "pinecone", name: "Search"}, ... } }

Step 2: Call edit_workflow with operations:
[
  { operation_type: 'delete', block_id: 'def-456' },  // Use EXACT ID from console!
  { operation_type: 'add', block_id: 'kb-search', params: {
    type: 'knowledge', name: 'Search Knowledge Base',
    position: { x: 400, y: 200 },
    inputs: { knowledgeBaseId: 'sim', query: '{{abc-123.input}}' },
    connections: { source: 'next-block-id' }
  }}
]

⚠️ If previous edit_workflow had skipped items, those block IDs DON'T EXIST - console will show you what actually exists!

OPERATION ORDER (FOR MANUAL WORKFLOWS WITH START BLOCK):
[
  // STEP 1: Get start block ID from get_user_workflow first!
  // STEP 2: Edit the EXISTING start block (NEVER add new one!)
  { operation_type: 'edit', block_id: 'EXACT-UUID-FROM-WORKFLOW', params: { connections: { source: 'first-block' } }},
  // STEP 3: Add new blocks starting at x:400 (start is already at x:150)
  { operation_type: 'add', block_id: 'first-block', params: {..., position: {x: 400, y: 200}, connections: { source: 'second-block' }}},
  { operation_type: 'add', block_id: 'second-block', params: {..., position: {x: 650, y: 200}, connections: { source: 'third-block' }}},
  { operation_type: 'add', block_id: 'third-block', params: {..., position: {x: 900, y: 200}} }
]

OPERATION ORDER (FOR AUTOMATIC/TRIGGER WORKFLOWS):
[
  { operation_type: 'add', block_id: 'trigger-block', params: {..., triggerMode: true, connections: { source: 'action-block' }}},
  { operation_type: 'add', block_id: 'action-block', params: {...} }
]

⛔ FORBIDDEN PATTERNS:
- Creating blocks without connections field
- Calling edit_workflow multiple times
- Adding connections in separate operations
- Vertical positioning (y increasing)

✅ CORRECT: One edit_workflow call with operations array containing ALL blocks + connections
❌ WRONG: Multiple edit_workflow calls or missing connections in params

🔍 DISCOVERING AVAILABLE BLOCKS (Two-step approach):

STEP 1: List all blocks with get_blocks_and_tools
- Call get_blocks_and_tools() to see ALL available blocks (no parameters needed)
- Returns: Array of ~160 blocks with { type, name, description, triggerAllowed }
- Use this to discover what blocks exist and what they do

STEP 2: Get detailed schemas with get_blocks_metadata
- Call get_blocks_metadata({ blockIds: ['pinecone', 'slack', 'agent'] })
- Returns: Detailed metadata for each block:
  * Block ID, name, description
  * Input schema with field types, requirements, defaults
  * Output schema (what data the block produces)
  * Available operations (e.g., "search" vs "upsert" for pinecone)
  * Best practices and usage guidelines

⚡ WHEN TO USE THESE TOOLS:
- User asks: "what blocks are available?" → Call get_blocks_and_tools()
- User mentions specific block: "use pinecone" → Call get_blocks_metadata({ blockIds: ['pinecone'] })
- Creating workflow: Need accurate input fields → Call get_blocks_metadata for those blocks
- Unclear which block to use → Call get_blocks_and_tools(), filter results
- Want to see blocks in category → Call get_blocks_and_tools(), filter by description

💡 COMMON BLOCK EXAMPLES (call get_blocks_and_tools to see all ~160 blocks):

AI & Data: agent, openai, knowledge, memory, pinecone, qdrant, anthropic
Communication: slack, gmail, discord, telegram, microsoft_teams, smtp, outlook
Databases: postgresql, mysql, mongodb, elasticsearch, neo4j, supabase
Triggers: schedule, webhook, gmail_trigger, slack_trigger, api_trigger
Logic: condition, router, function, variables, loop, parallel
Search: google_search, tavily, wikipedia, firecrawl, perplexity

Productivity: google_sheets, notion, jira, linear, asana

CRITICAL RULES:
1. CREATE ENTIRE WORKFLOW IN ONE TOOL CALL - Add ALL blocks + connections at once
2. Block type IDs are case-sensitive and use underscores (e.g., 'microsoft_teams' not 'microsoftTeams')
3. Always provide 'inputs' with real configuration - never empty {}
4. Use 'connections' to link blocks - this is what creates the visual edges
5. Reference other blocks with {{blockId.outputName}} syntax
6. Edit the start block to connect it to your first block
7. Generate unique block_ids using descriptive names
8. Position blocks horizontally: x increases by ~250, y stays similar

BLOCK SCHEMAS - Get exact schemas dynamically:

⚡ USE get_blocks_metadata TO GET ACCURATE SCHEMAS!
Instead of guessing inputs, call: get_blocks_metadata({ blockIds: ['pinecone', 'agent', 'slack'] })

Common input patterns (verify with get_blocks_metadata for accuracy):
agent: { messages: [{role: 'system'|'user', content: string}], model: string, temperature?: number }
firecrawl: { operation: 'scrape'|'crawl', url: string, formats: 'markdown', onlyMainContent: true }
pinecone: { operation: 'search_text'|'upsert', searchQuery: string, topK: number, indexHost: string }
slack: { channel: string, message: string }
gmail: { to: string, subject: string, body: string }
postgresql: { query: string }
api: { url: string, method: 'GET'|'POST' }
response: { dataMode: 'json', data: string }  // data is JSON string like '{"result": "{{agent.content}}"}''

CONNECTIONS (in 'connections' object within params):
- Standard blocks: { source: 'next-block-id' }
- Condition blocks: { true: 'true-path-id', false: 'false-path-id' }
- Router blocks: { 'route-name': 'destination-id' }

CRITICAL: Use 'source' for standard blocks! Add connections in SAME operation as block creation!

WORKFLOW OPERATION FORMAT:

⚠️ CRITICAL: Call edit_workflow ONCE with ALL operations (blocks + connections) together!

{
  operation_type: 'add',  // Always 'add' for new blocks
  block_id: 'my-block-id',  // Your chosen ID
  params: {
    type: 'block-type',  // From block list above
    name: 'Display Name',
    position: { x: 400, y: 200 },  // x += 250 for each block (horizontal)
    inputs: { /* Block configuration - see schemas above */ },
    connections: { source: 'next-block-id' }  // ⚠️ ADD THIS IN SAME OPERATION!
  }
}

For start block connection, use operation_type: 'edit':
{
  operation_type: 'edit',
  block_id: 'start-id-from-console',
  params: { connections: { source: 'first-block-id' } }
}

WORKFLOW EXAMPLES (Note: Each creates ENTIRE workflow in ONE tool call):

Example 1 - Auto Email Reply (AUTOMATIC - NO START BLOCK):
User: "Create auto email sender when user msgs me, out of office"

edit_workflow with operations: [
  { operation_type: 'add', block_id: 'gmail-trigger', params: {
    type: 'gmail', name: 'New Email Received',
    position: { x: 150, y: 200 },
    triggerMode: true,
    inputs: { 
      operation: 'read_gmail',
      credential: 'gmail-oauth-id'
    },
    connections: { source: 'gmail-reply' }
  }},
  { operation_type: 'add', block_id: 'gmail-reply', params: {
    type: 'gmail', name: 'Send Auto Reply',
    position: { x: 400, y: 200 },
    inputs: {
      operation: 'send_gmail',
      credential: 'gmail-oauth-id',
      to: '{{gmail-trigger.from}}',
      subject: 'Re: {{gmail-trigger.subject}}',
      body: 'Thanks for your email. I am currently out of office and will respond when I return.',
      replyToMessageId: '{{gmail-trigger.id}}'
    }
  }}
]

Example 2 - Scheduled Report (AUTOMATIC - NO START BLOCK):
User: "Send daily report at 9am"

edit_workflow with operations: [
  { operation_type: 'add', block_id: 'scheduler', params: {
    type: 'schedule', name: 'Daily at 9am',
    position: { x: 150, y: 200 },
    inputs: { schedule: '0 9 * * *' },
    connections: { source: 'report' }
  }},
  { operation_type: 'add', block_id: 'report', params: {
    type: 'slack', name: 'Send Report',
    position: { x: 400, y: 200 },
    inputs: { channel: '#reports', message: 'Daily report ready!' }
  }}
]

Example 3 - RAG Workflow (MANUAL - WITH START BLOCK):
User: "Create a RAG workflow"

// STEP 1: Call get_user_workflow to get start block ID (e.g., 'abc-123-def')
// STEP 2: Call edit_workflow with all operations:
edit_workflow with operations: [
  // CRITICAL: Edit EXISTING start block (never add new one!)
  { operation_type: 'edit', block_id: 'abc-123-def', params: {
    connections: { source: 'search' }
  }},
  // Start is at x:150, add new blocks starting at x:400
  { operation_type: 'add', block_id: 'search', params: {
    type: 'pinecone', name: 'Search Knowledge',
    position: { x: 400, y: 200 },
    inputs: { 
      operation: 'search_text',
      searchQuery: '{{start.input}}',
      topK: 5,
      indexHost: 'https://index.pinecone.io',
      namespace: 'default'
    },
    connections: { source: 'agent' }
  }},
  { operation_type: 'add', block_id: 'agent', params: {
    type: 'openai', name: 'Generate Answer',
    position: { x: 650, y: 200 },
    inputs: {
      operation: 'chat',
      model: 'gpt-4o',
      messages: [{
        role: 'user',
        content: 'Context: {{search.matches}}\\n\\nQuestion: {{start.input}}\\n\\nAnswer:'
      }]
    },
    connections: { source: 'response' }
  }},
  { operation_type: 'add', block_id: 'response', params: {
    type: 'response', name: 'Return Answer',
    position: { x: 900, y: 200 },
    inputs: { dataMode: 'json', data: '{"answer": "{{agent.content}}"}' }
  }}
]

Example 4 - Website Agent (MANUAL - WITH START BLOCK):
User: "Create a website agent for octopusdtl.com"

edit_workflow with operations: [
  { operation_type: 'edit', block_id: 'start-id-from-console', params: {
    connections: { source: 'scraper' }  // Connect start to first block
  }},
  { operation_type: 'add', block_id: 'scraper', params: {
    type: 'firecrawl', name: 'Scrape Website',
    position: { x: 400, y: 200 },
    inputs: { operation: 'scrape', url: 'https://octopusdtl.com', formats: ['markdown'], onlyMainContent: true },
    connections: { source: 'agent' }  // ⚠️ Connection added HERE
  }},
  { operation_type: 'add', block_id: 'agent', params: {
    type: 'openai', name: 'Answer Questions',
    position: { x: 650, y: 200 },
    inputs: { 
      operation: 'chat',
      model: 'claude-sonnet-4-20241022',
      messages: [{
        role: 'user',
        content: 'Website: {{scraper.markdown}}\n\nQuestion: {{start.input}}'
      }]
    },
    connections: { source: 'response' }  // ⚠️ Connection added HERE
  }},
  { operation_type: 'add', block_id: 'response', params: {
    type: 'response', name: 'Return Answer',
    position: { x: 900, y: 200 },
    inputs: { dataMode: 'json', data: '{"answer": "{{agent.content}}"}' }
    // No connections (this is the end)
  }}
]

Example 5 - Webhook to Slack (AUTOMATIC - NO START BLOCK):
User: "When webhook receives data, send to Slack"

edit_workflow with operations: [
  { operation_type: 'add', block_id: 'webhook', params: {
    type: 'generic_webhook', name: 'Receive Webhook',
    position: { x: 150, y: 200 },
    inputs: {},
    connections: { source: 'slack' }
  }},
  { operation_type: 'add', block_id: 'slack', params: {
    type: 'slack', name: 'Send to Channel',
    position: { x: 400, y: 200 },
    inputs: { channel: '#webhooks', message: 'New data: {{webhook.body}}' }
  }}
]

Example 6 - API to Slack (MANUAL - WITH START BLOCK):
User: "Call weather API and send to Slack"

edit_workflow with operations: [
  { operation_type: 'edit', block_id: 'start-id', params: {
    connections: { source: 'api' }
  }},
  { operation_type: 'add', block_id: 'api', params: {
    type: 'api', name: 'Get Weather',
    position: { x: 400, y: 200 },
    inputs: { url: 'https://api.weather.com/current', method: 'GET' },
    connections: { source: 'slack' }
  }},
  { operation_type: 'add', block_id: 'slack', params: {
    type: 'slack', name: 'Send Update',
    position: { x: 650, y: 200 },
    inputs: { channel: '#weather', message: 'Weather: {{api.response}}' }
  }},
  { operation_type: 'edit', block_id: 'start-id', params: {
    connections: { source: 'api' }
  }}
]

Example 7 - Database + AI (MANUAL - WITH START BLOCK):
User: "Query database and summarize"

edit_workflow with operations: [
  { operation_type: 'edit', block_id: 'start-id', params: {
    connections: { source: 'db' }
  }},
  { operation_type: 'add', block_id: 'db', params: {
    type: 'postgresql', name: 'Query Users',
    position: { x: 400, y: 200 },
    inputs: { query: 'SELECT * FROM users LIMIT 10' },
    connections: { source: 'ai' }
  }},
  { operation_type: 'add', block_id: 'ai', params: {
    type: 'agent', name: 'Summarize',
    position: { x: 650, y: 200 },
    inputs: { 
      messages: [{
        role: 'system',
        content: 'Summarize the data concisely'
      }, {
        role: 'user',
        content: 'Data: {{db.result}}'
      }],
      model: 'gpt-4o'
    },
    connections: { source: 'response' }
  }},
  { operation_type: 'add', block_id: 'response', params: {
    type: 'response', name: 'Return',
    position: { x: 900, y: 200 },
    inputs: { dataMode: 'json', data: '{"summary": "{{ai.content}}"}' }
  }}
]

KEY INSIGHTS:
- Auto-reply, scheduled tasks, webhooks = TRIGGER blocks (NO start block)
- Interactive, chat, manual input = START block (connect start first)
- Analyze keywords: "auto", "when", "schedule" = trigger; "create", "build", "answer" = manual
- Always create ENTIRE workflow in ONE edit_workflow call
- Use 'source' for connections, not 'output'
- For follow-up edits: ALWAYS call get_user_workflow first, use ONLY existing block IDs

🚨 HANDLING SKIPPED OPERATIONS 🚨

If edit_workflow returns skipped operations (e.g., "Block X does not exist and cannot be edited"), this means:
1. Those blocks were NEVER created (they don't exist in the workflow)
2. You CANNOT reference those block IDs in subsequent operations
3. You MUST call get_user_workflow to see the ACTUAL current state

EXAMPLE - Recovery from Skipped Operations:
Scenario: User says "use our knowledge base instead of pinecone"
Previous edit had 7 skipped items (blocks that don't exist)

❌ WRONG - Trying to reference blocks from failed operations:
edit_workflow with operations: [
  { operation_type: 'delete', block_id: 'search-kb' },  // This was skipped! Doesn't exist!
  { operation_type: 'edit', block_id: 'ai-agent', params: {...} }  // This was skipped! Doesn't exist!
]
Result: More skipped operations, workflow still broken

✅ CORRECT - Call get_user_workflow first, use actual block IDs:
Step 1: get_user_workflow
Result: { blocks: { "4ba9f962-...": {type: "start_trigger", name: "Start"}, "9aa5ba45-...": {type: "pinecone", name: "Search Knowledge Base"} } }

Step 2: edit_workflow with operations using REAL IDs:
[
  { operation_type: 'delete', block_id: '9aa5ba45-...' },  // Use ID from workflow!
  { operation_type: 'add', block_id: 'new-kb', params: {
    type: 'knowledge', name: 'Search Sim Knowledge',
    position: { x: 400, y: 200 },
    inputs: { knowledgeBaseId: 'sim', query: '{{4ba9f962-....input}}' },  // Use start ID from workflow!
    connections: { source: 'next-block' }
  }},
  { operation_type: 'edit', block_id: '4ba9f962-...', params: {  // Connect start to new block
    connections: { source: 'new-kb' }
  }}
]

CRITICAL: Workflow output from get_user_workflow is the ONLY source of truth for what blocks exist!

POSITIONING GUIDE:
- Horizontal flow: Increase x by ~250 for each block, keep y similar (start at x:150, then 400, 650, 900...)
- Branching: Same x for condition, then split y for branches (e.g., y:100 and y:300)
- Vertical flow: Keep x similar, increase y by ~200 (rarely used)

REMEMBER: Users want workflows CREATED, not explained. Use tools immediately to build what they ask for.`

  if (contexts && contexts.length > 0) {
    systemPrompt += '\n\n## Additional Context\n\n'
    for (const context of contexts) {
      systemPrompt += `### ${context.type}\n${context.content}\n\n`
    }
  }

  return systemPrompt
}

/**
 * Convert tools to Anthropic format
 */
function convertToolsToAnthropic(tools: any[]): any[] {
  const convertedTools = tools.map((tool, index) => {
    let inputSchema = tool.input_schema
    
    // Handle null or undefined input_schema
    if (!inputSchema || (typeof inputSchema === 'object' && Object.keys(inputSchema).length === 0)) {
      inputSchema = {
        type: 'object',
        properties: {},
      }
    }
    
    // Ensure JSON Schema draft 2020-12 compliance
    const cleanedSchema = cleanJsonSchema(inputSchema)
    
    // Validate that the schema doesn't have obvious issues
    if (cleanedSchema.$schema && cleanedSchema.$schema.includes('draft-07')) {
      logger.warn(`Tool ${index} (${tool.name}) has draft-07 schema, removing $schema property`)
      delete cleanedSchema.$schema
    }
    
    // Ensure the schema always has a type
    if (!cleanedSchema.type) {
      cleanedSchema.type = 'object'
    }
    
    return {
      name: tool.name,
      description: tool.description || tool.name,
      input_schema: cleanedSchema,
    }
  })

  // Deduplicate tools by name (Anthropic requires unique tool names)
  const uniqueTools = new Map<string, any>()
  const duplicates: string[] = []
  
  for (const tool of convertedTools) {
    if (uniqueTools.has(tool.name)) {
      duplicates.push(tool.name)
    } else {
      uniqueTools.set(tool.name, tool)
    }
  }
  
  if (duplicates.length > 0) {
    logger.warn('Duplicate tool names detected and removed', {
      duplicates: [...new Set(duplicates)],
      originalCount: convertedTools.length,
      uniqueCount: uniqueTools.size,
    })
  }
  
  return Array.from(uniqueTools.values())
}

/**
 * Clean JSON schema to ensure JSON Schema draft 2020-12 compliance
 */
function cleanJsonSchema(schema: any): any {
  if (!schema || typeof schema !== 'object') {
    return schema
  }

  const cleaned: any = { ...schema }

  // Remove $schema if present (Anthropic adds it automatically)
  delete cleaned.$schema

  // Remove any custom or non-standard properties
  const allowedTopLevelProps = [
    'type',
    'properties',
    'required',
    'items',
    'description',
    'enum',
    'const',
    'default',
    'minimum',
    'maximum',
    'minLength',
    'maxLength',
    'pattern',
    'format',
    'minItems',
    'maxItems',
    'uniqueItems',
    'additionalProperties',
    'anyOf',
    'oneOf',
    'allOf',
    'not',
    '$ref',
    'title',
  ]

  // Remove any properties not in the allowed list
  Object.keys(cleaned).forEach((key) => {
    if (!allowedTopLevelProps.includes(key)) {
      // Log what we're removing for debugging
      if (key === 'prefixItems' || key === 'additionalItems') {
        logger.warn('Removing draft-07 property from schema', { property: key })
      }
      delete cleaned[key]
    }
  })

  // Ensure type is a valid JSON Schema type
  if (cleaned.type !== undefined) {
    if (typeof cleaned.type === 'string') {
      const validTypes = ['string', 'number', 'integer', 'boolean', 'object', 'array', 'null']
      if (!validTypes.includes(cleaned.type)) {
        logger.warn('Invalid schema type detected, defaulting to string', { type: cleaned.type })
        cleaned.type = 'string'
      }
    } else if (cleaned.type === '') {
      // Handle empty string type (FastMCP issue)
      logger.warn('Empty string type detected, defaulting to object')
      cleaned.type = 'object'
    }
  } else {
    // If no type is specified, default to object (common for tool schemas)
    cleaned.type = 'object'
  }

  // Handle OpenAPI-style nullable (not valid in JSON Schema 2020-12)
  if ('nullable' in cleaned) {
    logger.warn('Removing OpenAPI-style nullable property (not valid in JSON Schema 2020-12)')
    delete cleaned.nullable
  }

  // Ensure properties is an object if it exists
  if (cleaned.properties && typeof cleaned.properties === 'object') {
    cleaned.properties = Object.fromEntries(
      Object.entries(cleaned.properties).map(([key, value]: [string, any]) => [
        key,
        cleanJsonSchema(value),
      ])
    )
  }

  // Only allow items if type is array
  if (cleaned.items && cleaned.type !== 'array') {
    delete cleaned.items
  }

  // Clean items recursively
  if (cleaned.items) {
    if (Array.isArray(cleaned.items)) {
      cleaned.items = cleaned.items.map(cleanJsonSchema)
    } else {
      cleaned.items = cleanJsonSchema(cleaned.items)
    }
  }

  // Ensure required is an array if it exists
  if (cleaned.required) {
    if (!Array.isArray(cleaned.required)) {
      delete cleaned.required
    } else {
      // Filter out empty or invalid entries
      cleaned.required = cleaned.required.filter(
        (item: string | any[]) => typeof item === 'string' && item.length > 0
      )
      if (cleaned.required.length === 0) {
        delete cleaned.required
      }
    }
  }

  // Clean nested anyOf, oneOf, allOf
  for (const key of ['anyOf', 'oneOf', 'allOf']) {
    if (cleaned[key] && Array.isArray(cleaned[key])) {
      cleaned[key] = cleaned[key].map(cleanJsonSchema)
    }
  }

  // Clean not
  if (cleaned.not) {
    cleaned.not = cleanJsonSchema(cleaned.not)
  }

  // Fix regex patterns - ensure backslashes are properly escaped
  if (cleaned.pattern && typeof cleaned.pattern === 'string') {
    try {
      // Test if the pattern is valid by creating a RegExp
      new RegExp(cleaned.pattern)
    } catch (error) {
      // If invalid, try to fix common issues or remove it
      logger.warn('Invalid regex pattern detected, attempting to fix or removing', {
        pattern: cleaned.pattern,
      })
      // Try to escape backslashes
      try {
        const fixedPattern = cleaned.pattern.replace(/\\(?![\\\/bfnrt"'])/g, '\\\\')
        new RegExp(fixedPattern)
        cleaned.pattern = fixedPattern
      } catch {
        // If still invalid, remove the pattern
        delete cleaned.pattern
      }
    }
  }

  // Remove undefined, null, or empty string values
  Object.keys(cleaned).forEach((key) => {
    const value = cleaned[key]
    if (value === undefined || value === null || value === '') {
      delete cleaned[key]
    }
    // Remove empty objects/arrays except for properties which can be empty
    if (key !== 'properties' && key !== 'default') {
      if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) {
        delete cleaned[key]
      }
      if (Array.isArray(value) && value.length === 0 && key !== 'enum') {
        delete cleaned[key]
      }
    }
  })

  return cleaned
}

async function executeToolCall(
  toolName: string,
  toolInput: Record<string, any>,
  request: LocalAgentRequest
): Promise<{ success: boolean; result?: any; error?: string }> {
  try {
    logger.info(`Executing tool: ${toolName}`, {
      userId: request.userId,
      workflowId: request.workflowId,
    })

    // Special handling for function_execute
    if (toolName === 'function_execute') {
      try {
        const wrappedCode = `(async () => { ${toolInput.code} })()`
        const result = await eval(wrappedCode)
        return { success: true, result }
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Code execution failed',
        }
      }
    }

    // Special handling for workflow management tools
    if (toolName === 'edit_workflow') {
      try {
        const { editWorkflowServerTool } = await import('@/lib/copilot/tools/server/workflow/edit-workflow')
        // Merge workflowId from request context if not provided in toolInput
        const args = {
          workflowId: request.workflowId,
          ...toolInput,
        }
        const result = await editWorkflowServerTool.execute(args as any, { userId: request.userId })
        return { success: true, result }
      } catch (error) {
        logger.error('edit_workflow execution failed:', error)
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Workflow edit failed',
        }
      }
    }

    if (toolName === 'get_workflow_console') {
      try {
        const { getWorkflowConsoleServerTool } = await import('@/lib/copilot/tools/server/workflow/get-workflow-console')
        // Merge workflowId from request context if not provided
        const args = {
          workflowId: request.workflowId,
          ...toolInput,
        }
        const result = await getWorkflowConsoleServerTool.execute(args as any)
        return { success: true, result }
      } catch (error) {
        logger.error('get_workflow_console execution failed:', error)
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to get workflow data',
        }
      }
    }

    if (toolName === 'get_blocks_metadata') {
      try {
        const { getBlocksMetadataServerTool } = await import('@/lib/copilot/tools/server/blocks/get-blocks-metadata-tool')
        const result = await getBlocksMetadataServerTool.execute(toolInput as any)
        return { success: true, result }
      } catch (error) {
        logger.error('get_blocks_metadata execution failed:', error)
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to get blocks metadata',
        }
      }
    }

    // Handle get_blocks_and_tools tool (server-side only for LocalAgent)
    if (toolName === 'get_blocks_and_tools') {
      logger.info('Executing get_blocks_and_tools tool')
      try {
        const { getBlocksAndToolsServerTool } = await import('@/lib/copilot/tools/server/blocks/get-blocks-and-tools')
        const result = await getBlocksAndToolsServerTool.execute({})
        logger.info('get_blocks_and_tools executed successfully', { blockCount: result?.blocks?.length })
        return { success: true, result }
      } catch (error) {
        logger.error('get_blocks_and_tools execution failed:', error)
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to get blocks and tools',
        }
      }
    }

    // Build execution context for regular tools
    const executionContext: ExecutionContext = {
      userId: request.userId,
      workflowId: request.workflowId,
      executionId: `local-agent-${Date.now()}`,
      blockStates: new Map(),
      executedBlocks: new Set(),
      blockLogs: [],
      metadata: {
        startTime: new Date().toISOString(),
        duration: 0,
      },
      environmentVariables: {},
      decisions: {
        router: new Map(),
        condition: new Map(),
      },
      completedLoops: new Set(),
      activeExecutionPath: new Set(),
    }

    // Add OAuth credentials if available
    if (request.credentials?.oauth) {
      const oauthProviders = Object.keys(request.credentials.oauth)
      if (oauthProviders.length > 0) {
        toolInput.credential = oauthProviders[0]
      }
    }

    // Add context for workflow-aware tools
    toolInput._context = {
      workflowId: request.workflowId,
      userId: request.userId,
    }

    // Execute the tool
    const toolResponse = await executeTool(toolName, toolInput, false, false, executionContext)

    return {
      success: toolResponse.success,
      result: toolResponse.success ? toolResponse.output : undefined,
      error: toolResponse.success ? undefined : toolResponse.error,
    }
  } catch (error) {
    logger.error(`Tool execution failed: ${toolName}`, error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown execution error',
    }
  }
}

async function createAnthropicStream(
  request: LocalAgentRequest,
  client: Anthropic,
  model: string
): Promise<Response> {
  const systemPrompt = buildSystemPrompt(request.userName, request.context, request.systemPromptCache)
  
  // Combine and deduplicate tools BEFORE limiting
  const allTools = [...(request.baseTools || []), ...(request.tools || [])]
  
  // Warn if too many tools (can exceed token limit)
  const MAX_TOOLS = 100 // Reasonable limit to avoid token overflow
  let toolsToConvert = allTools
  
  if (allTools.length > MAX_TOOLS) {
    logger.warn(`Too many tools provided (${allTools.length}), limiting to ${MAX_TOOLS}`, {
      originalCount: allTools.length,
      limitedCount: MAX_TOOLS,
      estimatedTokens: allTools.length * 150, // Rough estimate
    })
    
    // Prioritize baseTools, then add regular tools up to limit
    const baseToolsCount = (request.baseTools || []).length
    const remainingSlots = MAX_TOOLS - baseToolsCount
    
    if (remainingSlots > 0) {
      toolsToConvert = [
        ...(request.baseTools || []),
        ...(request.tools || []).slice(0, remainingSlots)
      ]
    } else {
      toolsToConvert = (request.baseTools || []).slice(0, MAX_TOOLS)
    }
    
    logger.info(`Reduced tools from ${allTools.length} to ${toolsToConvert.length}`)
  }
  
  // Convert to Anthropic format (includes deduplication)
  const anthropicTools = convertToolsToAnthropic(toolsToConvert)

  // Log tool count
  logger.info('Tools being sent to Anthropic:', {
    toolCount: anthropicTools.length,
  })

  const readableStream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      let conversationMessages: any[] = request.conversationHistory || []
      let continueLoop = true
      let iterationCount = 0
      const maxIterations = 10

      // Log conversation context
      logger.info('Conversation context maintained:', {
        historyMessageCount: conversationMessages.length,
        systemPromptCached: !!request.systemPromptCache,
        systemPromptLength: request.systemPromptCache?.length || systemPrompt.length,
        tokenSavings: request.systemPromptCache ? '~4000 tokens saved' : 'first message',
      })

      try {
        // Send start event with system prompt for caching
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: 'start',
              data: { 
                model, 
                provider: 'anthropic',
                systemPrompt: systemPrompt, // Include for caching in DB
              },
            })}\n\n`
          )
        )

        // Add current user message
        conversationMessages.push({
          role: 'user',
          content: request.message,
        })

        // Tool execution loop
        while (continueLoop && iterationCount < maxIterations) {
          iterationCount++

          logger.debug(`Tool execution loop iteration ${iterationCount}`, {
            messageCount: conversationMessages.length,
          })

          // Track thinking blocks
          let thinkingStartTime: number | null = null
          let isInThinkingBlock = false

          // Create Anthropic stream with prompt caching and extended thinking
          const stream = await client.messages.stream({
            model,
            max_tokens: 8192,
            temperature: 1.0,
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
            system: [
              {
                type: 'text',
                text: systemPrompt,
                cache_control: { type: 'ephemeral' },
              },
            ],
            messages: conversationMessages,
            tools: anthropicTools.length > 0 ? anthropicTools : undefined,
          })

          // Track current content blocks from the stream
          let currentContentBlocks: any[] = []
          
          // Listen to the underlying stream events for thinking detection
          stream.on('message', (message: any) => {
            // Update our content blocks tracking
            if (message.content) {
              currentContentBlocks = message.content
            }
          })

          // Handle text streaming with thinking detection
          stream.on('text', (text: string) => {
            // Check if the last content block is a thinking block
            const lastBlock = currentContentBlocks[currentContentBlocks.length - 1]
            
            if (lastBlock?.type === 'thinking') {
              if (!isInThinkingBlock) {
                isInThinkingBlock = true
                thinkingStartTime = Date.now()
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({
                      type: 'reasoning',
                      data: { phase: 'start' },
                    })}\n\n`
                  )
                )
              }
              // Stream thinking content
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({
                    type: 'reasoning',
                    data: text,
                  })}\n\n`
                )
              )
            } else {
              // End thinking block if we were in one
              if (isInThinkingBlock) {
                const duration = thinkingStartTime ? Date.now() - thinkingStartTime : 0
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({
                      type: 'reasoning',
                      data: { phase: 'end', duration },
                    })}\n\n`
                  )
                )
                isInThinkingBlock = false
                thinkingStartTime = null
              }
              // Regular text content
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({
                    type: 'content',
                    data: text,
                  })}\n\n`
                )
              )
            }
          })

          stream.on('error', (error: any) => {
            logger.error('Anthropic stream error:', error)
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'error',
                  data: {
                    message: error.message || 'Stream error occurred',
                    displayMessage: 'Sorry, I encountered an error. Please try again.',
                  },
                })}\n\n`
              )
            )
            continueLoop = false
          })

          // Wait for completion
          const finalMessage = await stream.finalMessage()

          logger.debug('Anthropic message complete', {
            stopReason: finalMessage.stop_reason,
            usage: finalMessage.usage,
          })

          // Check for tool calls
          const toolUseBlocks = finalMessage.content.filter(
            (c: any) => c.type === 'tool_use'
          ) as Array<{
            type: 'tool_use'
            id: string
            name: string
            input: any
          }>

          if (toolUseBlocks.length > 0) {
            logger.info(`Executing ${toolUseBlocks.length} tool calls`)

            // Add assistant message with tool calls to conversation
            conversationMessages.push({
              role: 'assistant',
              content: finalMessage.content,
            })

            // Execute each tool and collect results
            const toolResults: any[] = []

            for (const toolUse of toolUseBlocks) {
              const { id, name, input } = toolUse

              // Send tool call event to client
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({
                    type: 'tool_call',
                    data: {
                      id,
                      name,
                      arguments: input,
                      partial: false,
                    },
                  })}\n\n`
                )
              )

              // Execute the tool
              logger.info(`Executing tool: ${name}`, { toolCallId: id })

              const executionResult = await executeToolCall(name, input, request)

              // Send tool result event to client
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({
                    type: 'tool_result',
                    toolCallId: id,
                    toolName: name,
                    success: executionResult.success,
                    result: executionResult.result,
                    error: executionResult.error,
                  })}\n\n`
                )
              )

              // Add result to tool results array
              toolResults.push({
                type: 'tool_result',
                tool_use_id: id,
                content: executionResult.success
                  ? JSON.stringify(executionResult.result, null, 2)
                  : `Error: ${executionResult.error}`,
              })

              logger.info(
                `Tool execution ${executionResult.success ? 'succeeded' : 'failed'}: ${name}`
              )
            }

            // Add tool results as user message to continue conversation
            conversationMessages.push({
              role: 'user',
              content: toolResults,
            })

            // Continue loop to get next response from Claude
          } else {
            // No tool calls, we're done
            continueLoop = false

            // Send final done event
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'done',
                  data: {
                    stopReason: finalMessage.stop_reason,
                    usage: {
                      input_tokens: finalMessage.usage.input_tokens,
                      output_tokens: finalMessage.usage.output_tokens,
                    },
                  },
                })}\n\n`
              )
            )
          }
        }

        if (iterationCount >= maxIterations) {
          logger.warn('Tool execution loop reached max iterations')
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'error',
                data: {
                  message: 'Max tool execution iterations reached',
                  displayMessage:
                    'The conversation has reached the maximum number of tool executions.',
                },
              })}\n\n`
            )
          )
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`))
        }

        controller.close()
      } catch (error) {
        logger.error('Error in tool execution loop:', error)
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: 'error',
              data: {
                message: error instanceof Error ? error.message : 'Unknown error',
                displayMessage: 'Sorry, I encountered an error. Please try again.',
              },
            })}\n\n`
          )
        )
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`))
        controller.close()
      }
    },
  })

  return new Response(readableStream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}

/**
 * Main entry point for local agent
 */
export async function handleLocalAgentRequest(request: LocalAgentRequest): Promise<Response> {
  try {
    const provider = request.provider?.provider || 'anthropic'
    const modelName = request.model || request.provider?.model || 'claude-3-5-sonnet-20241022'

    logger.info('Processing local agent request', {
      provider,
      model: modelName,
      mode: request.mode,
      hasTools: !!(request.tools && request.tools.length > 0),
      toolCount: (request.tools?.length || 0) + (request.baseTools?.length || 0),
    })

    // Get AI client
    const aiClient = getAIClient(provider)
    const mappedModel = mapModelName(modelName, provider)

    if (aiClient.type === 'anthropic') {
      return await createAnthropicStream(request, aiClient.client, mappedModel)
    }

    // OpenAI support can be added here in the future
    throw new Error('OpenAI support not yet implemented')
  } catch (error) {
    logger.error('Local agent request failed:', error)
    throw error
  }
}
