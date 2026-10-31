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
    "task2_retrieval": (
        "The first image is a 2D engineering drawing of a CAD model. "
        "The following {n} images are rendered CAD models, labeled {labels}. "
        "Which rendered model does the drawing represent? "
        "Answer with exactly one letter."
    ),
    "task3_complexity": (
        "You are shown six views (front, back, left, right, top, bottom) of a CAD model. "
        "Classify the model as Simple or Complex based on its geometry. "
        "Answer with exactly one word: Simple or Complex."
    ),
}