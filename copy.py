import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import seaborn as sns
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
from datetime import datetime
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from streamlit_ace import st_ace
import subprocess
import uuid
import os
import google.generativeai as genai
import json
from streamlit_lottie import st_lottie  # Import the Lottie library
import requests  # Needed to fetch Lottie animations
# Load environment variables
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

conn = sqlite3.connect('user_profiles.db')
c = conn.cursor()

# Create table if it does not exist
c.execute('''CREATE TABLE IF NOT EXISTS profiles (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT,
             email TEXT,
             age INTEGER,
             bio TEXT,
             created_on TEXT
             )''')

conn.commit()

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


def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a QA chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def check_code_with_gemini(user_code, exercise_solution):
    """Check user code against the correct solution using Gemini API."""
    prompt = f"User's code:\n{user_code}\n\nExpected solution:\n{exercise_solution}\n\nEvaluate the user's code for correctness, identify the programming language, and suggest improvements if any."
    response = model.generate_content(prompt)
    return response



def generate_quiz(response):
    """Generate practice questions based on text chunks."""
    try:
        sample_chunk = random.choice(response)
        prompt = f"Generate 5 quiz questions based on the following text:\n{sample_chunk}"

        # Use generate_text to create quiz questions
        responsew = genai.generate_text(prompt=prompt)
        if responsew and hasattr(responsew, 'text'):
            questions = [q.strip() for q in responsew.text.split('\n') if q.strip()]  # Split the response into individual questions
            return questions
        else:
            st.error("The response from the API does not contain text.")
            return []
    except Exception as e:
        st.error(f"An error occurred while generating the quiz: {str(e)}")
        return []

def evaluate_answer(user_answer, correct_answer):
    """Evaluate if the user's answer is correct."""
    # Simple evaluation (could be improved with more complex logic)
    return user_answer.lower() in correct_answer.lower()


def display_resources():
    # Define course data with roadmaps and resources
    courses = {
        "Python Programming": {
            "intro": "Learn the basics of Python programming.",
            "roadmap": {
                "Beginner": ["Variables", "Data Types", "Control Flow", "Functions", "Modules"],
                "Intermediate": ["Object-Oriented Programming", "File Handling", "Error Handling", "Libraries"],
                "Advanced": ["Decorators", "Generators", "Concurrency", "Networking"]
            },
            "resources": {
                "Beginner": [
                    {"title": "Python Crash Course", "url": "https://www.youtube.com/watch?v=rfscVS0vtbw"},
                    {"title": "Python Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=khKv-8q7YmY"}
                ],
                "Intermediate": [
                    {"title": "Python OOP", "url": "https://www.youtube.com/watch?v=JeznW_7DlB0"},
                    {"title": "Python File Handling", "url": "https://www.youtube.com/watch?v=Uh2ebFW8OYM"}
                ],
                "Advanced": [
                    {"title": "Python Decorators", "url": "https://www.youtube.com/watch?v=FsAPt_9Bf3U"},
                    {"title": "Python Generators", "url": "https://www.youtube.com/watch?v=bD05uGo_sVI"}
                ]
            },
            "pdfs": {
                "Beginner": [
                    {"title": "Python Basics PDF", "url": "https://bugs.python.org/file47781/Tutorial_EDIT.pdf"}
                ],
                "Intermediate": [
                    {"title": "Intermediate Python Programming PDF",
                     "url": "https://example.com/intermediate_python.pdf"}
                ],
                "Advanced": [
                    {"title": "Advanced Python Techniques PDF", "url": "https://example.com/advanced_python.pdf"}
                ]
            }
        },
        "Machine Learning": {
            "intro": "Understand the fundamentals of machine learning and build predictive models.",
            "roadmap": {
                "Beginner": ["Introduction to Machine Learning", "Supervised Learning", "Unsupervised Learning"],
                "Intermediate": ["Feature Engineering", "Model Evaluation", "Hyperparameter Tuning"],
                "Advanced": ["Neural Networks", "Deep Learning", "Reinforcement Learning"]
            },
            "resources": {
                "Beginner": [
                    {"title": "Machine Learning by Andrew Ng", "url": "https://www.youtube.com/watch?v=aircAruvnKk"},
                    {"title": "Machine Learning Crash Course", "url": "https://www.youtube.com/watch?v=4b5W3wN6AdU"}
                ],
                "Intermediate": [
                    {"title": "Feature Engineering", "url": "https://www.youtube.com/watch?v=2H1e3cFJgRA"},
                    {"title": "Model Evaluation and Validation", "url": "https://www.youtube.com/watch?v=85dtiMz9tSo"}
                ],
                "Advanced": [
                    {"title": "Deep Learning Specialization", "url": "https://www.youtube.com/watch?v=Nj9kzF1GCqU"},
                    {"title": "Reinforcement Learning", "url": "https://www.youtube.com/watch?v=9z_ZzsmOosg"}
                ]
            },
            "pdfs": {
                "Beginner": [
                    {"title": "Introduction to Machine Learning PDF", "url": "https://example.com/ml_intro.pdf"}
                ],
                "Intermediate": [
                    {"title": "Feature Engineering PDF", "url": "https://example.com/feature_engineering.pdf"}
                ],
                "Advanced": [
                    {"title": "Deep Learning PDF", "url": "https://example.com/deep_learning.pdf"}
                ]
            }
        },
        "Data Science": {
            "intro": "Explore data science concepts and techniques for analyzing data.",
            "roadmap": {
                "Beginner": ["Data Analysis", "Data Visualization", "Statistics"],
                "Intermediate": ["Exploratory Data Analysis", "Hypothesis Testing", "Predictive Modeling"],
                "Advanced": ["Big Data", "Data Engineering", "Machine Learning for Data Science"]
            },
            "resources": {
                "Beginner": [
                    {"title": "Data Science Tutorial", "url": "https://www.youtube.com/watch?v=r-uOLxNrNk8"},
                    {"title": "Data Science Full Course", "url": "https://www.youtube.com/watch?v=Gv9_5P8zB9M"}
                ],
                "Intermediate": [
                    {"title": "Exploratory Data Analysis", "url": "https://www.youtube.com/watch?v=8XgZQyDGaUA"},
                    {"title": "Predictive Modeling", "url": "https://www.youtube.com/watch?v=s5QokQ7FiSA"}
                ],
                "Advanced": [
                    {"title": "Big Data Analysis", "url": "https://www.youtube.com/watch?v=n28IP3pNcYQ"},
                    {"title": "Data Engineering with Python", "url": "https://www.youtube.com/watch?v=H14bBuluwB8"}
                ]
            },
            "pdfs": {
                "Beginner": [
                    {"title": "Data Analysis PDF", "url": "https://example.com/data_analysis.pdf"}
                ],
                "Intermediate": [
                    {"title": "Predictive Modeling PDF", "url": "https://example.com/predictive_modeling.pdf"}
                ],
                "Advanced": [
                    {"title": "Big Data Analysis PDF", "url": "https://example.com/big_data_analysis.pdf"}
                ]
            }
        },
        "Web Development": {
            "intro": "Master the skills needed to build dynamic and responsive websites.",
            "roadmap": {
                "Beginner": ["HTML", "CSS", "JavaScript"],
                "Intermediate": ["Responsive Design", "APIs", "Frameworks (React, Angular)"],
                "Advanced": ["Server-Side Rendering", "Web Security", "Progressive Web Apps"]
            },
            "resources": {
                "Beginner": [
                    {"title": "Web Development Full Course", "url": "https://www.youtube.com/watch?v=UB1O30fR-EE"},
                    {"title": "Frontend Web Development Crash Course",
                     "url": "https://www.youtube.com/watch?v=3JluqTojuME"}
                ],
                "Intermediate": [
                    {"title": "Responsive Web Design", "url": "https://www.youtube.com/watch?v=srvUrASNj0s"},
                    {"title": "React Tutorial", "url": "https://www.youtube.com/watch?v=w7ejDZ8SWv8"}
                ],
                "Advanced": [
                    {"title": "Server-Side Rendering with Next.js",
                     "url": "https://www.youtube.com/watch?v=IkOVe40Sy0U"},
                    {"title": "Progressive Web Apps", "url": "https://www.youtube.com/watch?v=aCMbSyngXB4"}
                ]
            },
            "pdfs": {
                "Beginner": [
                    {"title": "HTML & CSS Basics PDF", "url": "https://example.com/html_css_basics.pdf"}
                ],
                "Intermediate": [
                    {"title": "Responsive Web Design PDF", "url": "https://example.com/responsive_web_design.pdf"}
                ],
                "Advanced": [
                    {"title": "Advanced Web Development PDF", "url": "https://example.com/advanced_web_development.pdf"}
                ]
            }
        },
        "Algorithms": {
            "intro": "Learn algorithmic techniques to solve computational problems efficiently.",
            "roadmap": {
                "Beginner": ["Basic Algorithms", "Sorting and Searching", "Recursion"],
                "Intermediate": ["Dynamic Programming", "Graph Algorithms", "Greedy Algorithms"],
                "Advanced": ["Advanced Data Structures", "Computational Complexity", "Approximation Algorithms"]
            },
            "resources": {
                "Beginner": [
                    {"title": "Algorithms Course", "url": "https://www.youtube.com/watch?v=8hly31xKli0"},
                    {"title": "Algorithms and Data Structures", "url": "https://www.youtube.com/watch?v=og7k5bbT4qQ"}
                ],
                "Intermediate": [
                    {"title": "Dynamic Programming", "url": "https://www.youtube.com/watch?v=oBt53YbR9Kk"},
                    {"title": "Graph Algorithms", "url": "https://www.youtube.com/watch?v=09_LlHjoEiY"}
                ],
                "Advanced": [
                    {"title": "Advanced Data Structures", "url": "https://www.youtube.com/watch?v=4OQeCuLYj-4"},
                    {"title": "Approximation Algorithms", "url": "https://www.youtube.com/watch?v=9BzWw1YEmKI"}
                ]
            },
            "pdfs": {
                "Beginner": [
                    {"title": "Introduction to Algorithms PDF",
                     "url": "https://example.com/introduction_algorithms.pdf"}
                ],
                "Intermediate": [
                    {"title": "Dynamic Programming Techniques PDF",
                     "url": "https://example.com/dynamic_programming.pdf"}
                ],
                "Advanced": [
                    {"title": "Advanced Data Structures PDF", "url": "https://example.com/advanced_data_structures.pdf"}
                ],
                "Advanced": [
                    {"title": "Advanced Data Structures PDF", "url": "https://example.com/advanced_data_structures.pdf"}
                ]
            }
        }
    }

    # Apply custom CSS for background and box styling
    st.markdown("""
            <style>
            .background {
                background-image: url('https://cdn.elearningindustry.com/wp-content/uploads/2023/06/Shutterstock_2287315871.jpg'); /* Replace with your image URL */
                background-size: cover;
                background-position: center;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
            }

            .resource-box {
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                background-color: #000; /* Changed to black */
                color: #fff;
                cursor: pointer;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="background"></div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.header("Learning Resources")

    # Search functionality
    search_query = st.text_input("Search for a topic to find related courses and YouTube videos:")

    if search_query:
        search_query = search_query.strip().lower()
        results_found = False

        for course, details in courses.items():
            if search_query in course.lower():
                st.subheader(f"Results for '{course}'")
                with st.expander(course, expanded=True):
                    st.markdown(f"**Introduction:** {details['intro']}")
                    st.markdown("**Roadmap:**")
                    for level, topics in details['roadmap'].items():
                        st.markdown(f"**{level}:** {', '.join(topics)}")
                        st.markdown("**Resources:**")
                        for resource in details['resources'][level]:
                            st.markdown(f"- [{resource['title']}]({resource['url']})")
                        st.markdown("**PDFs:**")
                        for pdf in details['pdfs'][level]:
                            st.markdown(f"- [{pdf['title']}]({pdf['url']})")
                    st.markdown("---")
                results_found = True

        if not results_found:
            st.write("No resources found for your search query.")

    # Display all courses
    st.subheader("Available Courses")
    for course, details in courses.items():
        with st.expander(course):
            st.markdown(f"**Introduction:** {details['intro']}")
            st.markdown("**Roadmap:**")
            for level, topics in details['roadmap'].items():
                st.markdown(f"**{level}:** {', '.join(topics)}")
                st.markdown("**Resources:**")
                for resource in details['resources'][level]:
                    st.markdown(f"- [{resource['title']}]({resource['url']})")
                st.markdown("**PDFs:**")
                for pdf in details['pdfs'][level]:
                    st.markdown(f"- [{pdf['title']}]({pdf['url']})")

    st.markdown("</div>", unsafe_allow_html=True)

    # Additional static resources
    st.subheader("Online Courses")
    st.markdown("""
            <div class="resource-box">
                Explore these free online courses to deepen your understanding.
                <ul>
                    <li><a href="https://www.coursera.org">Coursera Free Courses</a></li>
                    <li><a href="https://www.edx.org">edX Free Courses</a></li>
                    <li><a href="https://www.khanacademy.org">Khan Academy</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Tutorials")
    st.markdown("""
            <div class="resource-box">
                Find tutorials on various subjects.
                <ul>
                    <li><a href="https://www.freecodecamp.org">FreeCodeCamp</a></li>
                    <li><a href="https://www.w3schools.com">W3Schools</a></li>
                    <li><a href="https://developer.mozilla.org">MDN Web Docs</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Books & eBooks")
    st.markdown("""
            <div class="resource-box">
                Access free educational books and eBooks.
                <ul>
                    <li><a href="https://www.gutenberg.org">Project Gutenberg</a></li>
                    <li><a href="https://openlibrary.org">Open Library</a></li>
                    <li><a href="https://ocw.mit.edu">MIT OpenCourseWare</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Videos & Lectures")
    st.markdown("""
            <div class="resource-box">
                Watch these free video lectures and educational content.
                <ul>
                    <li><a href="https://www.ted.com/talks">TED Talks</a></li>
                    <li><a href="https://www.youtube.com">YouTube Educational Channels</a></li>
                    <li><a href="https://www.khanacademy.org">Khan Academy Videos</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Interactive Tools & Platforms")
    st.markdown("""
            <div class="resource-box">
                Explore interactive tools and platforms for hands-on learning.
                <ul>
                    <li><a href="https://www.codecademy.com">Codecademy</a></li>
                    <li><a href="https://www.datacamp.com">DataCamp</a></li>
                    <li><a href="https://leetcode.com">LeetCode</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

user_profiles = []
def get_user_profiles():
    """Retrieve user profiles from the database."""
    try:
        with sqlite3.connect('user_profiles.db') as conn:
            query = "SELECT name, email, age, bio, created_on FROM profiles"
            df = pd.read_sql_query(query, conn)
            df['created_on'] = pd.to_datetime(df['created_on'])
            return df
    except Exception as e:
        st.error(f"An error occurred while retrieving profiles: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def save_user_profile(name, email, age, bio, created_on):
    """Save the user profile data to the database."""
    try:
        with sqlite3.connect('user_profiles.db') as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO profiles (name, email, age, bio, created_on)
                         VALUES (?, ?, ?, ?, ?)''',
                      (name, email, age, bio, created_on))
            conn.commit()
            st.success("Profile saved successfully!")
    except Exception as e:
        st.error(f"An error occurred while saving the profile: {e}")

def display_user_profiles():
    """Display all user profiles."""
    st.title("User Profiles Dashboard")

    # Sidebar for creating user profiles
    st.sidebar.header("Create User Profile")
    with st.sidebar.form("profile_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        bio = st.text_area("Bio")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and email and age and bio:
                created_on = datetime.now()
                save_user_profile(name, email, age, bio, created_on)
                st.success("User profile created successfully!")
                st.experimental_rerun()  # Refresh the app to display the new profile
            else:
                st.error("Please fill out all fields.")

    # Retrieve user profiles from the database
    profile_data = get_user_profiles()

    st.subheader("User Profiles")
    if not profile_data.empty:
        # Display user profiles
        for _, profile in profile_data.iterrows():
            st.markdown(f"""
            **Name:** {profile['name']}
            **Email:** {profile['email']}
            **Age:** {profile['age']}
            **Bio:** {profile['bio']}
            **Created On:** {profile['created_on'].strftime('%Y-%m-%d %H:%M:%S')}
            """)
            st.markdown("---")

        # Plot user age distribution
        st.subheader("User Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(profile_data['age'], kde=True, ax=ax)
        ax.set_title("Age Distribution of Users")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Plot user registration timeline
        st.subheader("User Registration Timeline")
        fig, ax = plt.subplots()
        profile_data['created_on'].dt.to_period('M').value_counts().sort_index().plot(kind='line', ax=ax)
        ax.set_title("Number of Users Registered Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Registrations")
        st.pyplot(fig)

        # Additional metrics
        st.subheader("Additional Metrics")
        st.write(f"Total Users: {len(profile_data)}")
        st.write(f"Average Age: {profile_data['age'].mean():.2f} years")

    else:
        st.write("No user profiles available.")

def send_email(user_email, feedback):
    sender_email = "kniteesh577@gmail.com"
    receiver_email = "nithustarz@gmail.com"
    password = "workurassoff"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "New Feedback from User"

    body = f"User Email: {user_email}\n\nUser Feedback:\n\n{feedback}"
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Update with your SMTP server and port
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False






def main():
    st.set_page_config(page_title="Personalized Learning Platform", layout="wide")

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Choose a page", ["Home", "Profile", "Resources", "Coding", "Feedback"])

    if page == "Home":
        st.header("Welcome to the Personalized Learning Platform")
        st.write("Here you can ask questions about the PDF files and take practice quizzes.")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)

            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Extract the response text
            response_text = response.get("output_text", "Sorry, I couldn't find an answer.")

            # Display the answer
            st.write("**Reply:** ", response_text)

            if response_text != "Sorry, I couldn't find an answer.":
                st.subheader("Practice Quiz")

                # Generate initial quiz questions
                text_chunks = get_text_chunks(response_text)

                def generate_quiz(text_chunks):
                    quizzes = []
                    for chunk in text_chunks:
                        sentences = chunk.split('. ')
                        if len(sentences) > 1:
                            question = sentences[0] + "?"
                            correct_answer = sentences[1]
                            # Generate some random incorrect answers
                            incorrect_answers = generate_incorrect_answers(correct_answer, text_chunks)
                            options = [correct_answer] + incorrect_answers
                            random.shuffle(options)  # Shuffle the options

                            quiz = {
                                'question': question,
                                'options': options,
                                'correct_answer': correct_answer
                            }
                            quizzes.append(quiz)
                    return quizzes

                def generate_incorrect_answers(correct_answer, text_chunks):
                    incorrect_answers = []
                    for chunk in text_chunks:
                        sentences = chunk.split('. ')
                        for sentence in sentences:
                            if sentence and sentence != correct_answer and len(incorrect_answers) < 3:
                                incorrect_answers.append(sentence)
                                if len(incorrect_answers) == 3:
                                    break
                    return incorrect_answers

                def run_quiz(quizzes):
                    quiz_completed = False
                    user_answers = []

                    for idx, quiz in enumerate(quizzes, 1):
                        st.write(f"**Quiz {idx}:** {quiz['question']}")

                        options = quiz['options']
                        letters = ['A', 'B', 'C', 'D']
                        for option_idx, (letter, option) in enumerate(zip(letters, options), 1):
                            st.write(f"{letter}) {option}")

                        user_answer = st.selectbox(f"Your answer for Quiz {idx}", options=letters,
                                                   key=f"answer_quiz_{idx}")
                        if user_answer:
                            user_answers.append(user_answer)
                            correct_answer = quiz['correct_answer']
                            correct_option = letters[options.index(correct_answer)]
                            if user_answer == correct_option:
                                st.write("Correct! 🎉")
                            else:
                                st.write("Incorrect. Please try again.")

                    if len(user_answers) == len(quizzes):
                        quiz_completed = True

                    return quiz_completed, user_answers
                quizzes = generate_quiz(text_chunks)

                quiz_completed = False
                while not quiz_completed:
                    quiz_completed, user_answers = run_quiz(quizzes)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_docs = pdf_docs
                    st.success("Documents processed successfully!")

    elif page == "Profile":
        display_user_profiles()


    elif page == "Resources":
        display_resources()

    elif page == "Coding":
        problems_file = 'problems.json'  # Replace with the actual path to your problems file
        display_exercise_page(problems_file)

    elif page == "Feedback":
        st.header("Feedback")
        st.write("We'd love to hear your feedback!")

        user_email = st.text_input("Your Email Address")
        feedback = st.text_area("Your Feedback")

        if st.button("Submit Feedback"):
            if not user_email or not feedback:
                st.error("Please provide both your email address and feedback.")
            else:
                if send_email(user_email, feedback):
                    st.success("Thank you for your feedback! It has been sent to our team.")
                else:
                    st.error("There was an error sending your feedback. Please try again later.")



if __name__ == "__main__":
    main()