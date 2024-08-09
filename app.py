import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# App title
st.set_page_config(page_title="🧠💬 Lakshya: Mental Health Assistant")

@st.cache_resource
def load_model():
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter = "GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        add_bos_token=True,
        trust_remote_code=True,
        padding_side='left'
    )

    # Create peft model using base_model and finetuned adapter
    config = PeftConfig.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                 torch_dtype='auto')
    model = PeftModel.from_pretrained(model, adapter)

    return model, tokenizer

model, tokenizer = load_model()

# Sidebar for model parameters
with st.sidebar:
    st.title('🧠💬 Lakshya: Mental Health Assistant')
    st.subheader('Model Parameters')
    max_new_tokens = st.slider('Max New Tokens', min_value=50, max_value=1000, value=512, step=50)
    temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=0.7, step=0.1)

    st.markdown("""
    ### About
    Lakshya is a Mental Health Assistant here to provide support and information.
    Remember, it's not a substitute for professional help.
    If you're in crisis, please reach out to a mental health professional or emergency services.
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lakshya, your Mental Health Assistant. How can I support you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function for generating response
def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(conversation=messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors='pt').to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.batch_decode(output_ids.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return response

# User-provided prompt
if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Lakshya is thinking..."):
            response = generate_response(prompt)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.sidebar.button('Clear Chat History'):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lakshya, your Mental Health Assistant. How can I support you today?"}
    ]
    st.rerun()
