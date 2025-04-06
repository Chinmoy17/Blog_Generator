import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(
        model='G:\Projects 2025\Blog Generation\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )

    template = """
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Streamlit page config
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            background-color: black;
        }
        .stSelectbox>div>div>div>div {
            background-color: black;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📝 AI Blog Generator</h1>", unsafe_allow_html=True)
st.write("Generate tailored blogs with the power of LLaMA 2. Choose your audience and blog length!")

# Input form
with st.form(key="blog_form"):
    input_text = st.text_input("📌 Enter the Blog Topic", placeholder="e.g., AI in Healthcare")

    col1, col2 = st.columns(2)
    with col1:
        no_words = st.text_input("✍️ Number of Words", placeholder="e.g., 300")
    with col2:
        blog_style = st.selectbox(
            "👤 Writing for",
            ('Researchers', 'Data Scientist', 'Common People'),
            index=0
        )

    submit = st.form_submit_button("🚀 Generate Blog")

# Generate and display result
if submit:
    if input_text and no_words:
        with st.spinner('Generating your blog...'):
            output = getLLamaresponse(input_text, no_words, blog_style)
            st.markdown("### 🧾 Generated Blog")
            st.success(output)
    else:
        st.warning("Please fill out all fields before submitting!")

# Footer
st.markdown("---")
st.markdown("<small>Made with 💙 using Streamlit & LLaMA 2</small>", unsafe_allow_html=True)
