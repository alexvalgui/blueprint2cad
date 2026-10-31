PROMPTS = {
    "task1_verification": (
        "You are shown a rendered view of a CAD model and a 2D engineering drawing. "
        "The two images may show the part from different viewpoints. "
        "Do they depict the same part? "
        "Answer with exactly one word: Yes or No."
    ),
    "task1_verification_cot": (
        "You are shown a rendered view of a CAD model and a 2D engineering drawing. "
        "The two images may show the part from different viewpoints. "
        "Do they depict the same part? "
        "First, reason step by step about the part's overall silhouette, its proportions, "
        "and distinctive features such as holes, slots, or steps, and consider how the "
        "drawing would look from the render's viewpoint. "
        "Then, on a new final line, answer with exactly one word: Yes or No."
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
        "You are shown a dimensioned 2D engineering drawing of a part. "
        "From the dimension annotations, extract the overall width and overall height "
        "of the part as shown in this view. "
        "Answer with only the two numeric values as width,height — no units, no explanation."
    ),
    "task7_counting": (
        "You are shown a view of a CAD model. {question} "
        "Answer with only an integer, no explanation."
    ),
}