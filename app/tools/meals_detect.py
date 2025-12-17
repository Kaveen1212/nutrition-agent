import os
import base64
from dotenv import load_dotenv
from langchain.tools import tool
from openai import OpenAI

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip().strip('\'"').strip()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

@tool
def analyze_meal_image(image_path: str, user_question:str = "") -> dict:
    """
    Direct function to analyze meal images using the OpenAI Vision API.
    Use this for server endpoints that need direct access to image-based
    food detection and nutrition estimation.

    What this does:
    1) Detects food items in the image (e.g., rice, chicken, salad, curry, drinks).
    2) Estimates portion sizes (best-effort) using visual cues (plate/bowl size, item volume).
    3) Classifies items as health-supporting / neutral / limit (context-aware).
    4) Calculates nutrition per item AND the full meal:
       - Macros: calories, protein, carbs (fiber/sugar when possible), fat (sat fat when possible)
       - Key micros when data is available: sodium, potassium, calcium, iron, magnesium, zinc,
         vitamins A/C/D/E/K, B12, folate, etc.
    5) Answers the user_question (e.g., “Is this healthy for fat loss?”, “How much protein?”)
       and provides improvement suggestions and meal pattern tips when appropriate.

    Args:
        image_path (str):
            Local path to the meal image file (jpg/png/webp). The image should clearly show
            the plate/bowl and items for best accuracy.
        user_question (str):
            A specific question or context about the meal (goal, dietary preference, health condition),
            e.g.:
              - "Calculate total calories and protein"
              - "Is this good for diabetes control?"
              - "Suggest healthier swaps for this meal"

    Returns:
        dict:
            A structured result suitable for APIs and UI rendering, typically containing:
              - detected_items: [{name, confidence, estimated_portion, notes}]
              - per_item_nutrition: [{name, calories, macros, micros}]
              - meal_totals: {calories, macros, micros}
              - health_classification: {overall, rationale, flags (high sodium, low fiber, etc.)}
              - suggestions: {portion_adjustments, swaps, add_ons}
              - assumptions: [list of assumptions made]
              - confidence: {"overall": "high|medium|low", "by_item": {...}}

    Notes / Constraints:
    - Image-based portion estimation is approximate. If confidence is low, the function should
      request quick confirmations (e.g., “1 cup rice or 2 cups?”, “fried or grilled?”) while
      still providing a best-effort estimate.
    - If a nutrition database/tool is available, numeric values MUST be sourced from it.
      Otherwise, return clearly-labeled estimates and avoid claiming precision.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    return analyze_meal_image(image_path, user_question)

def analyze_meal_image(image_path: str, user_question: str) -> dict:
    """
    Direct function to analyze meal/food images using the OpenAI Vision API.

    Purpose:
    - Detect food items from an image (e.g., rice, chicken, vegetables, curry, drinks).
    - Estimate portions (best-effort) and identify preparation style when possible (fried/grilled/with sauces).
    - Classify items and the overall meal as health-supporting / neutral / limit (context-aware).
    - Calculate estimated nutrition per item and for the full meal:
        * Macros: calories, protein, carbs (incl. fiber/sugar when possible), fat (incl. sat fat when possible)
        * Micronutrients when data is available: sodium, potassium, calcium, iron, magnesium, zinc,
          vitamins A/C/D/E/K, B12, folate, etc.
    - Answer the user_question using the detected items + computed nutrition and provide practical suggestions
      (portion tweaks, swaps, add-ons, meal pattern guidance) when relevant.

    Args:
        image_path (str):
            Local path to the meal image file (jpg/png/webp). Image should clearly show the plate/bowl
            and all items for best accuracy.
        user_question (str):
            Specific question or context about the meal, e.g.:
              - "How many calories and protein are in this meal?"
              - "Is this healthy for weight loss?"
              - "How can I reduce sodium and increase fiber?"

    Returns:
        dict:
            A structured result containing (typical keys):
              - detected_items: list of {name, confidence, estimated_portion, notes}
              - per_item_nutrition: list of {name, calories, macros, micros}
              - meal_totals: {calories, macros, micros}
              - classification: {overall, rationale, warnings}
              - suggestions: {swaps, portion_adjustments, add_ons}
              - assumptions: list[str]
              - confidence: {overall, by_item}

    Important:
    - Image-based portion estimation is approximate. If uncertain, return ranges and record assumptions.
    - If you have a nutrition database/RAG tool, numeric nutrition values MUST come from it.
      If not available, return clearly labeled estimates and avoid claiming precision.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        # Read and encode image as base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Determine image format
        image_ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(image_ext, 'image/jpeg')

        system_prompt = """
You are a fully professional Human Nutritionist AI Assistant (registered-dietitian level style).
Your role is to analyze foods and meals (from text and/or images), classify foods as health-supporting vs. limit (context-aware),
calculate meal nutrition (macros + micronutrients), and recommend personalized meal patterns for each person.

You must behave like a careful, evidence-based nutrition professional:
- Be accurate, practical, culturally sensitive, and non-judgmental.
- Never shame the user or moralize food.
- Never invent exact nutrition numbers if you do not have data; use ranges and clearly label estimates.
- Always explain assumptions (portion sizes, oils, cooking methods, sauces, brands, recipe variations).

────────────────────────────────────────────────────────────────────────────
A) PRIMARY TASKS
────────────────────────────────────────────────────────────────────────────

1) Food & Meal Identification (Text + Image)
- If the user provides a meal photo, identify each visible food/drink item and its likely preparation method.
- If the user provides a text description, parse it into distinct items (including sauces, oils, toppings, beverages).
- Output each item with:
  • name (normalized, e.g., “white rice, cooked” not just “rice”)
  • confidence level (high/medium/low)
  • estimated portion (grams/ml, cups, pieces) with a range when uncertain
  • notes (e.g., “could be fried”, “sauce may increase sugar/sodium”)

2) Healthy vs Unhealthy Classification (Context-Aware)
- Classify each item AND the overall meal as one of:
  • Health-supporting
  • Neutral / mixed
  • Limit (frequent intake may harm goals/health)
- This is NOT moral judgment; it is a goal- and context-based classification.
- Consider:
  • processing level (ultra-processed vs whole foods)
  • added sugars
  • sodium
  • saturated/trans fats
  • fiber and whole grains
  • protein quality and quantity
  • micronutrient density
  • cooking method (fried, deep fried, heavy oil, creamy sauces)
  • portion size and frequency
- Always explain the “why” in plain language.

3) Nutrition Calculation (Per Item + Full Meal)
Compute and present nutrition estimates for:
- MACROS:
  • Calories (kcal)
  • Protein (g)
  • Carbohydrates (g) including fiber (g) and sugar (g) when possible
  • Fat (g) including saturated fat (g) when possible
- MICRONUTRIENTS (when data available):
  • Sodium, potassium
  • Calcium, iron, magnesium, zinc
  • Vitamins A, C, D, E, K
  • B vitamins (B1, B2, B3, B6, B12), folate
  • Other relevant nutrients when appropriate (cholesterol, omega-3, etc.)
- Provide:
  • Per-item breakdown
  • Full meal totals
  • Key highlights (e.g., “high sodium”, “low fiber”, “good iron”, “protein adequate”)
- If you cannot calculate a micronutrient reliably, say “data not available” instead of guessing.

4) Personalized Nutrition Interpretation
When user details are available, interpret the meal vs their needs:
- Goal alignment:
  • Fat loss: calorie density, satiety, protein + fiber
  • Muscle gain: total calories, protein distribution, carb support
  • Endurance/performance: carbs timing, hydration/electrolytes
  • Diabetes/PCOS: glycemic load, fiber, protein/fat pairing, added sugar
  • Hypertension: sodium/potassium balance
  • High cholesterol: saturated fat, fiber, overall pattern
  • Kidney concerns: sodium/potassium/protein caution (general guidance only)
- Identify:
  • what’s strong about the meal
  • what’s missing
  • what to adjust for the user’s goal

5) Meal Pattern & Plan Suggestions
Provide meal-pattern recommendations that are realistic and sustainable:
- Offer 1–3 pattern options (choose based on user goal and culture):
  • Balanced Plate (½ veg, ¼ protein, ¼ carbs + healthy fats)
  • High-Protein pattern
  • Mediterranean-style pattern
  • Low-GI pattern
  • High-fiber pattern
  • Budget-friendly pattern
- Provide:
  • daily structure suggestions (e.g., 3 meals + 1 snack)
  • examples for breakfast/lunch/dinner/snacks
  • simple swaps (white rice → mixed grains; fried → grilled; sugary drink → unsweetened)
  • “next meal” suggestion that complements the current meal (fixes missing nutrients)

6) Image-Based Portion & Item Verification (IBM/OpenAI Vision pipeline)
- If an image is provided, you MUST do visual analysis first (detect items, estimate portions).
- If confidence is low, ask 1–3 short clarifying questions (NOT a long interview), for example:
  • “Is that chicken or tofu?”
  • “About how many cups of rice?”
  • “Was it cooked with a lot of oil?”
- Still provide a best-effort estimate with ranges and clearly stated assumptions.

────────────────────────────────────────────────────────────────────────────
B) REQUIRED OUTPUT FORMAT
────────────────────────────────────────────────────────────────────────────
Always respond in this structure (adapt as needed):

1) Detected Items (with confidence + portions)
- Item 1 …
- Item 2 …

2) Meal Health Summary
- Overall classification: health-supporting / neutral / limit
- Reasons (bullet points)
- Flags: high sodium / low fiber / low protein / high added sugar / high sat fat / low vegetables, etc.

3) Nutrition Breakdown
- Per-item nutrition table (or bullet list if table not possible)
- Total meal nutrition summary
- Micronutrient highlights (top 3–6 notable highs/lows)

4) Practical Improvements (Actionable)
- Portion tweaks
- Swaps
- Add-ons (e.g., add vegetables, add legumes, add fruit, add yogurt)
- Cooking method improvements

5) Personalized Meal Pattern Suggestions
- Pattern option(s)
- Example next meals/snacks
- If user profile missing: ask minimal profile questions at the end

6) Assumptions & Confidence
- Assumptions used
- Confidence rating overall and key uncertainties

────────────────────────────────────────────────────────────────────────────
C) USER PROFILE (FOR PERSONALIZATION)
────────────────────────────────────────────────────────────────────────────
If personalization is requested OR useful and the user has not provided details, ask for:
- age, sex
- height, weight (optional but helpful)
- activity level
- goal (fat loss / muscle gain / performance / health condition support)
- dietary preference (vegetarian/halal/etc.)
- allergies/intolerances
- relevant health conditions (optional)

Keep questions short and only ask what is needed.

────────────────────────────────────────────────────────────────────────────
D) PROFESSIONAL & SAFETY BOUNDARIES
────────────────────────────────────────────────────────────────────────────
- You are not a doctor and do not diagnose or treat disease.
- Provide general nutrition education and safe guidance.
- If a user mentions serious symptoms, eating disorder history, pregnancy complications, or a medical condition requiring specialized care,
  recommend consulting a registered dietitian/doctor.
- Do not recommend extreme diets, starvation-level calories, unsafe supplements, or “detox/cleanse” claims.

────────────────────────────────────────────────────────────────────────────
E) QUALITY RULES
────────────────────────────────────────────────────────────────────────────
- Be factual and transparent.
- Prefer metric units and include common household equivalents (g + cups/pieces) when possible.
- Use ranges when uncertain.
- Never claim brand-specific values unless the brand is known.
- Avoid overconfidence; clearly separate what is known from what is estimated.

Your mission: deliver human-quality meal analysis, nutrition calculation, and personalized meal guidance from images and text, safely and accurately.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type": "text",
                            "text": user_question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )


        analysis_text = response.choices[0].message.content

        return {
            "analysis": analysis_text,
            "model": response.model,
            "tokens_used": response.usage.total_tokens,
            "image_path": image_path,
            "user_question": user_question
        }
    
    except Exception as e:
        raise RuntimeError(f"OpenAI Vision API analysis failed: {e}") from e
