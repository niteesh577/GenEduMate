import streamlit as st
import pandas as pd

# Sample user data for demonstration (replace with actual data)
user_data = {
    "user_id": [1, 2, 3, 4, 5],
    "preferences": ["Python, Machine Learning", "Web Development", "Data Science", "Algorithms", "Mathematics"],
    "completed_courses": [
        ["Python Programming", "Machine Learning"],
        ["Web Development"],
        ["Data Science", "Algorithms"],
        ["Algorithms"],
        ["Mathematics"]
    ],
    "quiz_scores": [
        [80, 85],
        [90],
        [75, 80],
        [70],
        [85]
    ]
}

# Convert user data into a DataFrame
user_df = pd.DataFrame(user_data)

# Sample courses data (replace with actual data)
courses = {
    "Python Programming": {
        "resources": {
            "beginner": [{"title": "Python Basics", "url": "https://example.com/python-basics"}],
            "intermediate": [{"title": "Intermediate Python", "url": "https://example.com/intermediate-python"}],
            "advanced": [{"title": "Advanced Python", "url": "https://example.com/advanced-python"}]
        },
        "roadmap": {
            "beginner": ["Variables", "Data Types", "Loops"],
            "intermediate": ["Functions", "Modules", "File Handling"],
            "advanced": ["Decorators", "Context Managers", "Concurrency"]
        }
    },
    # Add other courses here...
}


def get_user_profile(user_id):
    """Get user profile based on user_id."""
    user_profile = user_df[user_df["user_id"] == user_id].iloc[0]
    return user_profile


def recommend_courses(user_profile):
    """Generate course recommendations based on user preferences and completed courses."""
    recommendations = []
    preferences = user_profile["preferences"].split(", ")

    # Course recommendations based on user preferences
    for preference in preferences:
        for course in courses.keys():
            if preference.lower() in course.lower() and course not in user_profile["completed_courses"]:
                recommendations.append(course)

    return recommendations


def recommend_resources(user_profile):
    """Generate resource recommendations based on user profile."""
    recommendations = []
    preferences = user_profile["preferences"].split(", ")

    # Resource recommendations based on user preferences
    for preference in preferences:
        for course, details in courses.items():
            if preference.lower() in course.lower():
                for level, resources in details["resources"].items():
                    recommendations.extend(resources)

    return recommendations


def recommend_quizzes(user_profile):
    """Generate quiz recommendations based on user profile and quiz scores."""
    recommendations = []
    preferences = user_profile["preferences"].split(", ")

    # Quiz recommendations based on user preferences and performance
    for preference in preferences:
        for course, details in courses.items():
            if preference.lower() in course.lower():
                for level, topics in details["roadmap"].items():
                    for topic in topics:
                        if topic not in user_profile["completed_courses"]:
                            recommendations.append(f"Quiz on {topic} in {course}")

    return recommendations


def display_recommendations():
    """Display personalized recommendations to the user."""
    st.header("Personalized Recommendations")

    # User ID selection
    user_id = st.selectbox("Select your User ID", options=user_df["user_id"].tolist())
    user_profile = get_user_profile(user_id)

    # Recommendations
    st.subheader("Recommended Courses")
    courses_recommendations = recommend_courses(user_profile)
    if courses_recommendations:
        for course in courses_recommendations:
            st.write(f"- {course}")
    else:
        st.write("No new courses to recommend at this time.")

    st.subheader("Recommended Resources")
    resources_recommendations = recommend_resources(user_profile)
    if resources_recommendations:
        for resource in resources_recommendations:
            st.write(f"- {resource['title']} ({resource['url']})")
    else:
        st.write("No new resources to recommend at this time.")

    st.subheader("Recommended Quizzes")
    quizzes_recommendations = recommend_quizzes(user_profile)
    if quizzes_recommendations:
        for quiz in quizzes_recommendations:
            st.write(f"- {quiz}")
    else:
        st.write("No new quizzes to recommend at this time.")


def main():
    st.set_page_config(page_title="Personalized Learning Platform", layout="wide")

    display_recommendations()


if __name__ == "__main__":
    main()