PROMPTS = {
    "task1_verification": (
        "You are shown a rendered view of a CAD model and a 2D engineering drawing. "
        "Does this 2D drawing correspond to this rendered view of the model? "
        "Answer with exactly one word: Yes or No."
    ),
    "task2_view_id": (
        "The first image is the FRONT view of a CAD model. "
        "The second image is another view of the same model. "
        "Which view is the second image? Options: "
        "(A) back, (B) left, (C) right, (D) top, (E) bottom. "
        "Answer with exactly one letter: A, B, C, D or E."
    ),
    "task3_retrieval": (
        "The first image is a 2D engineering drawing of a CAD model. "
        "The following {n} images are rendered CAD models, labeled {labels}. "
        "Which rendered model does the drawing represent? "
        "Answer with exactly one letter."
    ),
    "task4_complexity": (
        "You are shown six views (front, back, left, right, top, bottom) of a CAD model. "
        "Classify the model as Simple or Complex based on its geometry. "
        "Answer with exactly one word: Simple or Complex."
    ),
    "task5_third_view": (
        "The first image is the FRONT view of a CAD model and the second image is its TOP view. "
        "The following four images, labeled A, B, C and D, are candidate RIGHT-side views. "
        "Which one is the correct right-side view of the model? "
        "Answer with exactly one letter: A, B, C or D."
    ),
    "task6_dimension": (
        "You are shown a dimensioned 2D engineering drawing. {question} "
        "Answer with only the numeric value, no units, no explanation."
    ),
    "task7_counting": (
        "You are shown a view of a CAD model. {question} "
        "Answer with only an integer, no explanation."
    ),
}
