import streamlit as st
import time
import json  # FIXED: Added missing import
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APITimeoutError
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="🏥 OpenAI Medical AI Agent",
    page_icon="⚕️",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        'current_task': None,
        'task_history': [],
        'tools_used': [],
        'reasoning_steps': []
    }
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Agent Tools Definition (SIMULATED)
AGENT_TOOLS = {
    "web_search": {
        "description": "Search web for current medical information (simulated)",
        "parameters": ["query"],
        "use_case": "When needing up-to-date medical information"
    },
    "medical_knowledge_base": {
        "description": "Query medical knowledge base (simulated)",
        "parameters": ["condition", "query_type"],
        "use_case": "For established medical facts"
    },
    "image_analysis": {
        "description": "Analyze medical images (simulated - educational only)",
        "parameters": ["image_data", "image_type"],
        "use_case": "For educational image discussion (NOT clinical diagnosis)"
    },
    "symptom_checker": {
        "description": "Analyze symptom patterns (simulated)",
        "parameters": ["symptoms", "patient_context"],
        "use_case": "For differential diagnosis education"
    },
    "drug_interaction_checker": {
        "description": "Check drug interactions (simulated)",
        "parameters": ["medications"],
        "use_case": "For medication safety education"
    },
    "clinical_calculator": {
        "description": "Calculate medical scores (simulated)",
        "parameters": ["calculator_type", "parameters"],
        "use_case": "For clinical risk scoring education"
    }
}

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg", width=120)
    st.header("⚙️ OpenAI Configuration")
    
    # API key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Get your key at https://platform.openai.com/api-keys"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        if not api_key.startswith("sk-"):
            st.warning("⚠️ Valid OpenAI keys start with 'sk-...'")
        else:
            st.success("✅ Key accepted")
    
    st.divider()
    
    # Model selection
    model = st.selectbox(
        "Model Selection",
        [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo"
        ],
        index=0,
        help="gpt-4o-mini: Best value • gpt-4o: Best quality • gpt-3.5-turbo: Budget"
    )
    
    pricing = {
        "gpt-4o-mini": "$0.15 / 1M input tokens",
        "gpt-4o": "$5.00 / 1M input tokens",
        "gpt-3.5-turbo": "$0.50 / 1M input tokens"
    }
    st.caption(f"💰 Pricing: {pricing[model]}")
    
    st.divider()
    
    st.header("🎯 Agent Settings")
    
    agent_mode = st.radio(
        "Agent Mode",
        ["Simple (1 API call)", "Standard (2-3 calls)", "Advanced (4-5 calls)"],
        index=0,
        help="Simple mode recommended to avoid rate limits"
    )
    
    mode_map = {
        "Simple (1 API call)": 1,
        "Standard (2-3 calls)": 3,
        "Advanced (4-5 calls)": 5
    }
    max_iterations = mode_map[agent_mode]
    
    enable_tools = st.multiselect(
        "Enabled Tools (SIMULATED)",
        list(AGENT_TOOLS.keys()),
        default=["medical_knowledge_base", "symptom_checker"],
        help="⚠️ ALL TOOLS ARE SIMULATED FOR EDUCATION"
    )
    
    delay_between_calls = st.slider(
        "Delay Between Calls (seconds)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="Prevents rate limits"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower = more factual"
    )
    
    st.divider()
    
    # Connection test
    if api_key and api_key.startswith("sk-"):
        if st.button("✅ Check OpenAI Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                try:
                    client = OpenAI(api_key=api_key)
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    if completion.choices[0].message.content:
                        st.success("✅ Connection successful!")
                except AuthenticationError:
                    st.error("❌ Invalid API key")
                except RateLimitError:
                    st.error("❌ Rate limit exceeded")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:150]}")
    
    st.divider()
    
    st.header("📊 Agent Status")
    if st.session_state.agent_state['current_task']:
        st.info(f"🔄 Current: {st.session_state.agent_state['current_task']}")
    else:
        st.success("✅ Agent Ready")
    
    st.metric("Tools Used", len(st.session_state.agent_state['tools_used']))
    st.metric("Reasoning Steps", len(st.session_state.agent_state['reasoning_steps']))
    
    st.divider()
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.agent_state = {
            'current_task': None,
            'task_history': [],
            'tools_used': [],
            'reasoning_steps': []
        }
        st.rerun()

# Helper Functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64"""
    buffered = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

def call_openai_with_retry(messages: List[Dict], system_prompt: str, api_key: str, 
                           model: str, temp: float = 0.2, delay: float = 1.0, 
                           max_retries: int = 3) -> str:
    """Call OpenAI API with retry logic"""
    client = OpenAI(api_key=api_key)
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
    
    for attempt in range(max_retries):
        try:
            # Only add delay on retries, not first attempt
            if attempt > 0:
                wait_time = delay * (2 ** attempt)
                st.warning(f"⏳ Retry {attempt}/{max_retries} in {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=1500,
                timeout=30.0
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            if attempt == max_retries - 1:
                return "❌ Rate limit exceeded. Wait 60s and try again."
            continue
            
        except AuthenticationError:
            return "❌ Invalid API key. Get new key at platform.openai.com"
            
        except APITimeoutError:
            return "❌ Request timed out. Try again in 10s."
            
        except APIError as e:
            return f"❌ API error: {str(e)[:200]}"
            
        except Exception as e:
            return f"❌ Error: {str(e)[:250]}"
    
    return "❌ Max retries exceeded."

# Simulated Tool Implementations
def execute_tool(tool_name: str, parameters: Dict) -> str:
    """Execute simulated tool"""
    disclaimer = "\n\n⚠️ **SIMULATION**: Educational purposes only. Not real medical data."
    
    if tool_name == "web_search":
        query = parameters.get("query", "medical condition")
        return f"🔍 Simulated search for '{query}':\n• Recent guidelines suggest...\n• Studies indicate...{disclaimer}"
    
    elif tool_name == "medical_knowledge_base":
        condition = parameters.get("condition", "general")
        return f"📚 Knowledge base - {condition}:\n• Definition: ...\n• Symptoms: ...\n• Treatment: ...{disclaimer}"
    
    elif tool_name == "image_analysis":
        return f"🔬 Image analysis:\n⚠️ AI CANNOT diagnose from images.\nEducational observations only.{disclaimer}"
    
    elif tool_name == "symptom_checker":
        symptoms = parameters.get("symptoms", [])
        return f"🩺 Symptom analysis:\n⚠️ NOT A DIAGNOSIS\nEducational considerations only.{disclaimer}"
    
    elif tool_name == "drug_interaction_checker":
        return f"💊 Drug interaction check:\n⚠️ SIMULATION ONLY\nConsult pharmacist for real checks.{disclaimer}"
    
    elif tool_name == "clinical_calculator":
        return f"🧮 Clinical calculator:\n⚠️ EDUCATIONAL ONLY\nUse validated tools in practice.{disclaimer}"
    
    return f"✅ Tool executed (simulated){disclaimer}"

# Agent Class
class MedicalAgent:
    """Medical agent with safety guardrails"""
    
    def __init__(self, api_key: str, model: str, enabled_tools: List[str], 
                 max_iter: int, temp: float, delay: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.enabled_tools = enabled_tools
        self.max_iterations = max_iter
        self.temperature = temp
        self.delay = delay
        self.reasoning_trace = []
        self.tools_used = []
        
    def plan_and_execute(self, user_query: str, context: List[Dict], 
                        image_data: Optional[str] = None) -> Dict:
        """Main agent execution"""
        if self.max_iterations <= 1:
            return self._simple_execution(user_query, image_data)
        return self._full_execution(user_query, context, image_data)
    
    def _simple_execution(self, user_query: str, image_data: Optional[str] = None) -> Dict:
        """Single API call execution"""
        content = [{
            "type": "text",
            "text": f"""You are a medical education assistant.

CRITICAL SAFETY RULES:
1. NEVER diagnose, treat, or prescribe
2. ALWAYS include: "⚠️ Educational information ONLY. NOT medical advice. Consult a physician."
3. State confidence level (High/Medium/Low)
4. For emergencies: "Call emergency services immediately"

Query: {user_query}"""
        }]
        
        if image_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "low"}
            })
            content[0]["text"] += "\n\n⚠️ Image for educational discussion only. NOT diagnosis."
        
        self.reasoning_trace.append({
            "step": "Direct Analysis",
            "content": "Single API call analysis",
            "timestamp": datetime.now().isoformat()
        })
        
        answer = call_openai_with_retry(
            [{"role": "user", "content": content}],
            self._get_system_prompt("synthesizer"),
            self.api_key, self.model, self.temperature, self.delay
        )
        
        return {
            "answer": answer,
            "reasoning_trace": self.reasoning_trace,
            "tools_used": [],
            "iterations": 1
        }
    
    def _full_execution(self, user_query: str, context: List[Dict], 
                       image_data: Optional[str] = None) -> Dict:
        """Multi-step execution with tools"""
        # Planning phase
        plan = call_openai_with_retry(
            [{"role": "user", "content": self._create_planning_prompt(user_query, image_data)}],
            self._get_system_prompt("planner"),
            self.api_key, self.model, self.temperature, self.delay
        )
        
        self.reasoning_trace.append({
            "step": "Planning",
            "content": plan,
            "timestamp": datetime.now().isoformat()
        })
        
        # Execution phase
        tool_results = []
        for iteration in range(self.max_iterations):
            action = call_openai_with_retry(
                [{"role": "user", "content": self._create_action_prompt(user_query, plan, tool_results, iteration)}],
                self._get_system_prompt("executor"),
                self.api_key, self.model, self.temperature, self.delay
            )
            
            self.reasoning_trace.append({
                "step": f"Action {iteration + 1}",
                "content": action,
                "timestamp": datetime.now().isoformat()
            })
            
            if "FINAL_ANSWER" in action:
                break
            
            tool_call = self._parse_tool_call(action)
            if tool_call:
                tool_name, params = tool_call
                if tool_name in self.enabled_tools:
                    result = execute_tool(tool_name, params)
                    tool_results.append({"tool": tool_name, "params": params, "result": result})
                    self.tools_used.append(tool_name)
        
        # Synthesis phase
        final_answer = call_openai_with_retry(
            [{"role": "user", "content": self._create_synthesis_prompt(user_query, plan, tool_results, image_data)}],
            self._get_system_prompt("synthesizer"),
            self.api_key, self.model, self.temperature, self.delay
        )
        
        return {
            "answer": final_answer,
            "reasoning_trace": self.reasoning_trace,
            "tools_used": list(set(self.tools_used)),
            "iterations": len(tool_results) + 1
        }
    
    def _get_system_prompt(self, role: str) -> str:
        """Get system prompt for role"""
        base = """Medical education assistant. NEVER diagnose/treat. ALWAYS include disclaimer."""
        
        prompts = {
            "planner": f"{base}\nCreate 3-4 step plan prioritizing safety.",
            "executor": f"{base}\nDecide next action or FINAL_ANSWER.",
            "synthesizer": f"{base}\nSynthesize with confidence level and disclaimer."
        }
        return prompts.get(role, base)
    
    def _create_planning_prompt(self, query: str, image_data: Optional[str]) -> str:
        prompt = f"Query: {query}\n\nAvailable tools:\n"
        for tool in self.enabled_tools:
            prompt += f"- {tool}: {AGENT_TOOLS[tool]['description']}\n"
        prompt += "\nCreate minimal 3-4 step plan prioritizing safety."
        return prompt
    
    def _create_action_prompt(self, query: str, plan: str, results: List[Dict], iteration: int) -> str:
        prompt = f"Query: {query}\nPlan: {plan[:200]}...\n\nCompleted: {len(results)} actions\n"
        prompt += f"Iteration {iteration + 1}/{self.max_iterations}\n"
        prompt += "Decide: FINAL_ANSWER or choose ONE tool to use next."
        return prompt
    
    def _create_synthesis_prompt(self, query: str, plan: str, results: List[Dict], image_data: Optional[str]) -> str:
        prompt = f"Query: {query}\n\nGathered information:\n"
        for i, r in enumerate(results, 1):
            prompt += f"{i}. {r['tool']}: {r['result'][:200]}...\n"
        prompt += "\nSynthesize with: 1) Answer 2) Confidence 3) BOLD disclaimer 4) Next steps"
        return prompt
    
    def _parse_tool_call(self, decision: str) -> Optional[Tuple[str, Dict]]:
        """Parse tool call from decision"""
        if "TOOL:" not in decision:
            return None
            
        lines = decision.split('\n')
        tool_name = None
        params = {}
        
        for line in lines:
            if "TOOL:" in line:
                tool_name = line.split("TOOL:")[1].strip()
            elif "PARAMS:" in line:
                try:
                    params_str = line.split("PARAMS:")[1].strip()
                    params = json.loads(params_str) if params_str.startswith("{") else {"query": params_str}
                except:
                    params = {"query": tool_name or "query"}
        
        return (tool_name, params) if tool_name in self.enabled_tools else None

# Main UI
st.title("🏥 OpenAI Medical AI Agent")
st.markdown("*Powered by GPT-4o with safety guardrails*")

st.warning("""
⚠️ **MEDICAL DISCLAIMER**: Educational information ONLY  
❌ NOT medical advice, diagnosis, or treatment  
✅ ALWAYS consult a licensed physician  
🚨 Emergencies: Call 911 immediately
""")

if not st.session_state.api_key:
    st.info("🔑 Get OpenAI API key at platform.openai.com/api-keys")

st.markdown("---")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], width=300)
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "reasoning" in message:
            with st.expander("⚡ View Reasoning", expanded=False):
                for step in message["reasoning"]:
                    st.markdown(f"**{step['step']}**")
                    content = step.get('content', step.get('result', ''))
                    st.text(content[:600])
                    st.caption(step['timestamp'])
                    st.divider()

# Input
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.chat_input("Ask a medical question...")
with col2:
    uploaded_image = st.file_uploader("📸", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Process
if user_input or uploaded_image:
    if not st.session_state.api_key or not st.session_state.api_key.startswith("sk-"):
        st.error("❌ Valid API key required")
        st.stop()
    
    user_msg = {"role": "user", "content": user_input or "Medical query"}
    image_data = None
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            image_data = encode_image_to_base64(image)
            user_msg["image"] = image
        except Exception as e:
            st.error(f"❌ Image error: {str(e)}")
            st.stop()
    
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        if uploaded_image:
            st.image(image, width=300)
        st.markdown(user_input or "Medical query")
    
    with st.chat_message("assistant"):
        st.session_state.agent_state['current_task'] = user_input or "Query"
        
        try:
            agent = MedicalAgent(
                api_key=st.session_state.api_key,
                model=model,
                enabled_tools=enable_tools,
                max_iter=max_iterations,
                temp=temperature,
                delay=delay_between_calls
            )
            
            start = time.time()
            result = agent.plan_and_execute(
                user_query=user_input or "Medical inquiry",
                context=st.session_state.conversation_history,
                image_data=image_data
            )
            elapsed = time.time() - start
            
            st.markdown(result["answer"])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ Time", f"{elapsed:.1f}s")
            col2.metric("Tools", len(result["tools_used"]) or "None")
            col3.metric("Steps", len(result["reasoning_trace"]))
            col4.metric("API Calls", result["iterations"])
            
            if result["tools_used"]:
                st.info(f"🔧 Tools: {', '.join(result['tools_used'])}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "reasoning": result["reasoning_trace"]
            })
            
            st.session_state.agent_state['tools_used'].extend(result["tools_used"])
            st.session_state.agent_state['reasoning_steps'].extend(result["reasoning_trace"])
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:250]}")
        finally:
            st.session_state.agent_state['current_task'] = None

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #dc2626; padding: 15px; background: #fff2f2; border-radius: 8px;'>
    <strong>⚠️ CRITICAL NOTICE</strong><br>
    SIMULATED tools • Educational only • NOT real medical databases<br>
    NEVER use for diagnosis/treatment • Consult licensed providers<br>
    Emergencies: Call 911 immediately
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align: center; color: gray; margin-top: 10px;'>
    ⚕️ Powered by OpenAI • Model: {model} • Tools SIMULATED
</div>
""", unsafe_allow_html=True)
