import streamlit as st
from dotenv import load_dotenv
from streamlit_ace import st_ace
import subprocess
import uuid
import os
import google.generativeai as genai
import json
from streamlit_lottie import st_lottie  # Import the Lottie library
import requests  # Needed to fetch Lottie animations

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Define available programming languages and their extensions for subprocess
languages = {
    'Python': 'python3',
    'JavaScript': 'node',
    'Java': 'java',
    'C++': 'g++',
}

# Functions to run code in different languages
def run_code(code, lang, filename):
    try:
        if lang == 'python3':
            process = subprocess.Popen(['python3', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
        elif lang == 'node':
            with open(f'{filename}.js', 'w') as f:
                f.write(code)
            process = subprocess.Popen(['node', f'{filename}.js'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
            os.remove(f'{filename}.js')
        elif lang == 'java':
            with open(f'{filename}.java', 'w') as f:
                f.write(code)
            subprocess.run(['javac', f'{filename}.java'], check=True)
            process = subprocess.Popen(['java', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.remove(f'{filename}.java')
            os.remove(f'{filename}.class')
        elif lang == 'g++':
            with open(f'{filename}.cpp', 'w') as f:
                f.write(code)
            # Compile the C++ code
            compilation = subprocess.run(['g++', f'{filename}.cpp', '-o', filename], stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
            if compilation.returncode != 0:
                return compilation.stderr, None
            # Run the compiled executable
            process = subprocess.Popen([f'./{filename}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.remove(f'{filename}.cpp')
            os.remove(f'{filename}')
        else:
            return "Execution not supported for this language", None

        # Get the output and error from the running process
        output, error = process.communicate()
        return output, error
    except Exception as e:
        return str(e), None

# Function to fetch Lottie animations from a URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load problems from a JSON file
def load_problems(filename):
    with open(filename, 'r') as file:
        problems = json.load(file)
    return problems

# Function to provide AI feedback
def get_ai_feedback(user_code, exercise_description):
    """Get AI feedback on the user's code."""
    prompt = (f"User's code:\n{user_code}\n\nProblem statement:\n{exercise_description}\n\n"
              "Evaluate the user's code for correctness and provide suggestions for improvement without giving away the solution.")
    response = model.generate_content(prompt)
    return response.text

def display_exercise_page(problems_file):
    """Display interactive exercises with feedback and progress tracking."""
    st.title("Interactive Coding Exercises")

    # Load problems
    problems = load_problems(problems_file)

    # Sidebar for problem selection
    st.sidebar.title('Select Problem')
    problem_names = [problem['title'] for problem in problems]
    selected_problem_title = st.sidebar.selectbox('Choose a problem', options=problem_names)

    # Sidebar for language selection
    st.sidebar.title('Select Language')
    selected_language = st.sidebar.selectbox('Choose a language', options=list(languages.keys()))

    # Get the selected problem
    problem = next(p for p in problems if p['title'] == selected_problem_title)
    language_mode = languages.get(selected_language, 'python3')

    st.subheader('Problem Description')
    st.write(problem["description"])

    # Lottie animation for explaining the problem
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_h4fnybmu.json"  # Example Lottie URL
    lottie_animation = load_lottie_url(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, height=300)

    # Code editor
    code = st_ace(language=language_mode, theme='monokai', height=400)

    # Button to run code
    if st.button('Run Code'):
        unique_id = uuid.uuid4().hex
        filename = f'{unique_id}'

        output, error = run_code(code, language_mode, filename)

        st.write('### Output')
        if output:
            st.code(output)
        if error:
            st.error(error)

        # AI feedback
        feedback = get_ai_feedback(code, problem["description"])
        st.write("### AI Feedback")
        st.write(feedback)

        # Reward system (simplified)
        if error is None and output:  # Simplified example condition
            st.success("Congratulations! You've solved the exercise. Here's your reward: 🏆")

    st.sidebar.title('Feedback Chatbot')
    st.write("Chatbot: Need help? Ask me anything about the exercise.")
    # For actual chatbot integration, you would connect to a GenAI model here.

if __name__ == "__main__":
    # Example problems file path
    problems_file = 'problems.json'  # Replace with the actual path to your problems file
    display_exercise_page(problems_file)