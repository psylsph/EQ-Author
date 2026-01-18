"""Prompt building utilities for story planning and chapter generation."""

from typing import List, Optional
from .constants import CHAPTER_WORD_TARGET, CHAPTER_MIN_WORDS


def build_step1_prompt(story_prompt: str) -> str:
    """Build the initial brainstorming and reflection prompt."""
    return (
        "Initial Writing Prompt:\n"
        f"{story_prompt}\n--\n"
        "Your task is to create a writing plan for this prompt. The scope will be a long format novel; do not assume a fixed number of chapters yet. "
        "Your plan should be comprehensive and in this format:\n"
        "# Brainstorming\n"
        "<Brainstorm ideas for characters, plot, tone, story beats, and possible pacing. The purpose of brainstorming is to cast a wide net of ideas, not to settle on any specific direction. "
        "Think about various ways you could take the prompt.>\n"
        "# Reflection\n"
        "<Reflect out loud on what works and doesn't work in these ideas. The purpose of this reflection is to narrow in on what you think will work best to make a piece that is a. compelling, and b. fits the prompt requirements. "
        "You are not making any decisions just yet, just reflecting.>\n"
        "Finally, propose the ideal number of chapters for this long format novel based on the prompt and your analysis.\n"
        "Output a single line at the end in this exact format so it can be parsed reliably:\n"
        "CHAPTER_COUNT: <integer>\n"
    )


def build_followup_prompts(n_chapters: int) -> List[str]:
    """Build prompts for steps 2-5 after chapter count is known."""
    p2 = (
        "Perfect. Now with the brainstorming complete, we will flesh out our characters. Lets go through each of our main characters:\n"
        "- Write about their background, personality, idiosyncrasies, flaws. Be specific and come up with examples to anchor & ground the character's profile (both core and trivial)\n"
        "- Briefly describe their physicality: appearance, how they carry themselves, express, interact with the world.\n"
        "- Concisely detail their motives, allegiances and existing relationships. Think from the perspective of the character as a real breathing thinking feeling individual in this world.\n"
        "- Write a couple quotes of flavour dialogue / internal monologue from the character to experiment with their voice.\n"
        "Output like this:\n"
        "# Character 1 name\n<character exploration>\n# Character 2 name\n<character exploration>\n etc"
    )

    p3 = (
        "Great now let's continue with planning the long format novel. Output in this format:\n"
        "# Intention\n"
        "<State your formulated intentions for the piece, synthesised from the the parts of the brainstorming session that worked, and avoiding the parts that didn't. "
        "Be explicit about the choices you have made about plot, voice, stylistic choices, things you intend to aim for & avoid.>\n"
        "# Chapter Planning\n"
        f"<Write a brief chapter plan for all {n_chapters} chapters.>"
    )

    p4 = (
        "With a view to making the writing more human, discuss how a human might approach this particular piece (given the original prompt). "
        "Discuss telltale LLM approaches to writing (generally) and ways they might not serve this particular piece. For example, common LLM failings are to write safely, or to always wrap things up with a bow, or trying to write impressively at the expense of readability. "
        "Then do a deep dive on the intention & plan, critiquing ways it might be falling into typical LLM tropes & pitfalls. Brainstorm ideas to make it more human. Be comprehensive. We aren't doing any rewriting of the plan yet, just critique & brainstorming."
    )

    p5 = (
        "Ok now with these considerations in mind, formulate the final plan for the a human like, compelling short piece in {n_chapters} chapters. Bear in mind the constraints of the piece (each chapter is just {word_target} words). "
        "Above all things, the plan must serve the original prompt. We will use the same format as before:\n"
        "# Intention\n"
        "<State your formulated intentions for the piece, synthesised from the the parts of the brainstorming session that worked, and avoiding the parts that didn't. "
        "Be explicit about the choices you have made about plot, voice, stylistic choices, things you intend to aim for & avoid.>\n"
        "# Chapter Planning\n"
        f"<Write a brief chapter plan for all {n_chapters} chapters.>"
    ).format(n_chapters=n_chapters, word_target=CHAPTER_WORD_TARGET)

    return [p2, p3, p4, p5]


def chapter_prompt(i: int, previous_chapter_ending: Optional[str] = None) -> str:
    """Build prompt for generating chapter i."""
    chapter_label = "Prologue" if i == 0 else f"Chapter {i}"
    
    # Use 1500 for Prologue, 2250 for regular chapters
    target_words = 1500 if i == 0 else 2250
    
    base_intro = (
        f"Write {chapter_label} of the story, following the approved plan and prior chapters.\n"
        f"- Produce at least {target_words} words of narrative prose.\n"
        "- Count only the words in your final story text; do not include planning notes or analysis.\n"
        f"- Output only the polished chapter text (you may open with a '{chapter_label}' heading if that matches the style), and do not mention the word count or include any commentary.\n"
        "\nCRITICAL INSTRUCTIONS FOR ACCURACY:\n"
        "1. **Adhere to the Plan**: Stick strictly to the events planned for THIS specific chapter. Do not rush ahead to future plot points.\n"
        "2. **Character Accuracy**: Refer constantly to the Character Profiles (Step 2) in your context. Ensure all details—especially ranks, titles, and relationships—are exactly as defined. Do not hallucinate promotions or demotions unless they are part of the story events.\n"
        "3. **Scope Boundaries**: Maintain the scope of this chapter. If the plan says the chapter ends at event X, do not go beyond event X.\n"
    )

    if i == 0 or i == 1:
        return "Great. Let's begin the story.\n" + base_intro

    overlap_instructions = (
        "\nIMPORTANT: Begin this chapter at a natural point after the previous chapter ended.\n"
        "- DO NOT repeat or restate events that just occurred at the end of the previous chapter.\n"
        "- Assume the reader has just finished the previous chapter and start with new content.\n"
        "- If referencing the previous chapter's ending, do so subtly without retelling the events.\n"
        "- Create a clear boundary between chapters - each chapter should feel distinct.\n"
        "- Think of this as a TV show episode that picks up after the previous one ended, not by replaying the last scene.\n"
    )

    if previous_chapter_ending:
        overlap_instructions += (
            f"\nNOTE: The previous chapter ended with: {previous_chapter_ending}\n"
            "Start this chapter AFTER this moment, not by repeating it."
        )

    return "Continue with the next installment.\n" + base_intro + overlap_instructions
