import ollama
from pydantic import BaseModel
import json
from typing import Literal

class ProductData(BaseModel):
    name: str
    gender: Literal["male", "female"]
    age: int
    weight: float
    height: float
    bmi: float | None = None
    frequentlypurchasedproduct: str | None = None
    recommended_product: str

class ProductDataList(BaseModel):
    Products: list[ProductData]

def calculate_bmi(weight_lbs, height_inches):
    """Calculate Product Score using weight in pounds and height in inches."""
    return round((weight_lbs * 703) / (height_inches ** 2), 1)

def extract_Product_data(text):
    """Extract last purchased Product information using LLM and return structured data."""
    
    # Create a prompt that specifies the expected JSON structure
    prompt = f"""Extract last purchased Product information from this shooping history text and recommend next product to be purchased and return it as JSON with the following structure:
    - name (string)
    - gender ("male" or "female")
    - age (number)
    - weight (number in pounds)
    - height (number in inches)
    - bmi (number)
    - frequentlypurchasedproduct (string)
    - recommended_product (string)

Text: {text}"""
    
    # Get response from LLM
    response = ollama.chat(
        model='llama3.2',
        messages=[{
            'role': 'user', 
            'content': prompt
        }],
        format='json',
        options={"temperature": 0}
    )

    try:
        # Validate response using Pydantic
        Product_data = ProductData.model_validate_json(response.message.content)
        
        # Calculate BMI of the customer based on the data provided
        if not Product_data.bmi:
           Product_data.bmi = calculate_bmi(Product_data.weight, Product_data.height)
        
        return Product_data.model_dump()
        
    except Exception as e:
        print(f"Error processing response: {e}")
        return None

def main():
    # Test Products
    ProductPurchaseHistory = [
        """John R. Whitaker, a 52-year-old male, stands 5'10" (70 inches) tall and weighs 198
        lbs. Mr. Whitaker has a history of frequent online purchases of skincare and beauty products, 
        which began in his mid-40s, and recently started exploring high-end anti-aging creams and serums. 
        He also regularly buys hair care products, attributing this habit to concerns about maintaining a 
        youthful appearance. Over the past six months, John has shifted towards eco-friendly and cruelty-free 
        brands, reflecting growing interest in sustainable beauty options. Additionally, he frequently purchases 
        fragrances, with a preference for seasonal and limited-edition collections, which he admits are indulgences 
        that bring him joy. Despite occasional overspending, Mr. Whitaker remains enthusiastic about experimenting
        with new products but admits to inconsistent use of certain items and difficulty sticking to a defined
        skincare routinet.""",
        """Emily J. Rivera has a pattern of purchasing mid-range to luxury cosmetic products, especially during sales 
        and promotions. She works as a schoolteacher and notes that her shopping habits often increase during periods of 
        stress, which she initially attributed to retail therapy. A closer look at her purchasing patterns reveals a preference 
        for skincare items targeting hydration and anti-aging, along with borderline frequent purchases of new makeup releases. 
        Emily is 41 years old, 5'5" (65 inches) tall, and weighs 172 lbs. She also frequently buys hand creams and cuticle oils, 
        expressing concern over dry, stiff hands in the mornings, which her favorite beauty consultant suggests could benefit 
        from specialized products. Emily’s busy schedule and irregular skincare routine have made it challenging to achieve 
        the consistent results she desires, though she remains enthusiastic about exploring new beauty trends and improving her 
        regimen with expert advice.""",
        """Karen L. Thompson, a 38-year-old female, is 5'4" (64 inches) tall and weighs 162 lbs. She has a history of frequent 
        purchases of skincare products for sensitive skin and deluxe hair care items, both of which have become more targeted 
        and premium over the past year. Karen also exhibits a strong preference for aromatherapy candles and bath products, 
        which she uses to combat chronic fatigue and stress. Recently, she has explored anti-aging creams and serums, leading 
        her favorite beauty advisor to suggest personalized recommendations for her evolving needs. She reports frequent impulse 
        buys of limited-edition makeup collections and occasional splurges on high-end perfumes, which she attributes to work-related 
        stress relief. Karen’s shopping habits are influenced by her demanding job as a paralegal, where long hours leave her 
        seeking convenient online beauty deals. Her late-night browsing often leads to spontaneous purchases, impacting both her 
        budget and her sleep schedule."""
    ]

    for i, Product_text in enumerate(ProductPurchaseHistory, 1):
        print(f"\nProcessing Product {i}:")
        result = extract_Product_data(Product_text)
        if result:
            print(json.dumps(result, indent=2))

if __name__ == '__main__':
   main() 
