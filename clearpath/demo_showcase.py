"""Fictional campus names mapped to real OULAD enrolment ids for the demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Course:
    id: str
    name: str
    code: str
    term: str
    dept: str


@dataclass(frozen=True)
class Student:
    id: str
    name: str
    course_id: str
    enrolment_id: str
    year: str
    status: str
    blurb: str


COURSES: tuple[Course, ...] = (
    Course(
        id="se101",
        name="Software Engineering Practice",
        code="SE101",
        term="Spring 2024",
        dept="Computing",
    ),
    Course(
        id="ds201",
        name="Data Science Foundations",
        code="DS201",
        term="Spring 2024",
        dept="Computing",
    ),
    Course(
        id="ba110",
        name="Digital Business Analytics",
        code="BA110",
        term="Spring 2024",
        dept="Business",
    ),
)

STUDENTS: tuple[Student, ...] = (
    Student(
        id="amina",
        name="Amina Khan",
        course_id="se101",
        enrolment_id="100561|DDD|2014J",
        year="Year 2",
        status="At risk",
        blurb="Low recent engagement; needs a clear next step.",
    ),
    Student(
        id="daniel",
        name="Daniel Okonkwo",
        course_id="se101",
        enrolment_id="102221|DDD|2014J",
        year="Year 2",
        status="On track",
        blurb="Steady progress; keep momentum with targeted resources.",
    ),
    Student(
        id="maya",
        name="Maya Chen",
        course_id="se101",
        enrolment_id="626630|DDD|2014J",
        year="Year 2",
        status="Catching up",
        blurb="Returning after a quiet fortnight.",
    ),
    Student(
        id="priya",
        name="Priya Sharma",
        course_id="ds201",
        enrolment_id="1006742|FFF|2014B",
        year="Year 1",
        status="Quiz-focused",
        blurb="Assessment week; quizzes dominate the path.",
    ),
    Student(
        id="leo",
        name="Leo Martins",
        course_id="ds201",
        enrolment_id="105939|FFF|2014J",
        year="Year 1",
        status="On track",
        blurb="Strong quiz performance with content follow-ups.",
    ),
    Student(
        id="noah",
        name="Noah Berg",
        course_id="ds201",
        enrolment_id="102854|FFF|2014J",
        year="Year 1",
        status="Distinction path",
        blurb="High achiever; stretch materials still help.",
    ),
    Student(
        id="sofia",
        name="Sofia Alvarez",
        course_id="ba110",
        enrolment_id="103496|BBB|2014J",
        year="Year 1",
        status="On track",
        blurb="Balanced mix of content and checks.",
    ),
    Student(
        id="jordan",
        name="Jordan Lee",
        course_id="ba110",
        enrolment_id="106793|BBB|2014J",
        year="Year 1",
        status="At risk",
        blurb="Needs lighter, high-signal next activities.",
    ),
    Student(
        id="ethan",
        name="Ethan Brooks",
        course_id="ba110",
        enrolment_id="677877|BBB|2014J",
        year="Year 1",
        status="Assessment week",
        blurb="Headed into quizzes; sequence matters.",
    ),
)

# Friendly titles keyed by activity id.
_TITLE_BANK: dict[str, tuple[str, ...]] = {
    "forumng": (
        "Discussion: Project standup reflections",
        "Forum: Ask the tutor this week",
        "Peer review thread: draft feedback",
        "Discussion: Common bugs & fixes",
        "Forum: Group coordination board",
    ),
    "resource": (
        "Reading: Lecture slides (PDF)",
        "Handout: Worked examples",
        "Reading: Case study brief",
        "File: Template checklist",
        "Notes: Summary sheet",
        "Reading: Spec excerpt",
        "Handout: Marking criteria",
        "File: Dataset description",
    ),
    "oucontent": (
        "Unit: Core concepts",
        "Lesson: Guided walkthrough",
        "Topic: Applied examples",
        "Unit: Practice problems",
        "Lesson: Key definitions",
        "Topic: Mini case",
        "Unit: Checkpoint summary",
        "Lesson: Extension material",
    ),
    "quiz": (
        "Quiz: Weekly knowledge check",
        "Quiz: Formative practice",
        "Quiz: Concept mastery",
        "Quiz: Rapid review",
        "Quiz: Pre-lab check",
    ),
    "externalquiz": (
        "External quiz: Practice set",
        "External quiz: Timed drill",
    ),
    "subpage": (
        "Section: Overview",
        "Section: Resources hub",
        "Section: This week's plan",
        "Section: Support links",
        "Section: Submission guide",
    ),
    "questionnaire": (
        "Survey: Learning reflection",
        "Survey: Weekly pulse check",
    ),
    "url": (
        "Link: External reading",
        "Link: Tool documentation",
    ),
    "glossary": (
        "Glossary: Key terms",
    ),
    "oucollaborate": (
        "Live session: Tutorial room",
    ),
    "ouwiki": (
        "Wiki: Shared notes",
    ),
    "page": (
        "Page: Announcement",
    ),
    "dataplus": (
        "Data lab: Explore the dataset",
    ),
    "homepage": (
        "Course homepage",
    ),
}


def course_by_id(course_id: str) -> Course | None:
    for c in COURSES:
        if c.id == course_id:
            return c
    return None


def student_by_id(student_id: str) -> Student | None:
    for s in STUDENTS:
        if s.id == student_id:
            return s
    return None


def student_by_enrolment(enrolment_id: str) -> Student | None:
    for s in STUDENTS:
        if s.enrolment_id == enrolment_id:
            return s
    return None


def activity_title(activity_type: str, id_site: int) -> str:
    bank = _TITLE_BANK.get(str(activity_type), ())
    if not bank:
        nice = str(activity_type).replace("_", " ").strip().title() or "Activity"
        return f"{nice} #{id_site}"
    return bank[int(id_site) % len(bank)]


def showcase_payload() -> dict:
    courses = [
        {
            "id": c.id,
            "name": c.name,
            "code": c.code,
            "term": c.term,
            "dept": c.dept,
        }
        for c in COURSES
    ]
    students = []
    for s in STUDENTS:
        course = course_by_id(s.course_id)
        students.append(
            {
                "id": s.id,
                "name": s.name,
                "course_id": s.course_id,
                "course_name": course.name if course else "",
                "course_code": course.code if course else "",
                "enrolment_id": s.enrolment_id,
                "year": s.year,
                "status": s.status,
                "blurb": s.blurb,
                "initials": "".join(p[0] for p in s.name.split()[:2]).upper(),
            }
        )
    return {"courses": courses, "students": students}
