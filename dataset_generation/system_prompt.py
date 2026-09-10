SYSTEM_PROMPT = """
We are modeling human-like forgetting of a sentence (memory item) by progressively removing episodic and fine-grained details over discrete time steps (t).
Your task to generate a list of sentences that represent the progressive and monotonic detail attenuation of a given sentence (memory item).

Follow these CORE guidelines strictly:
1. Progressive Detail Attenuation: As t increases, remove fine-grained details first, like numbers, dates, timestamps, dosages, specific locations, specific objects, proper nouns, and precise modifiers. For descriptive modifiers, remove the adjectives modifying objects, and then remove the adjectives of subject and others. Core semantic structure (who did what to whom) should persist longer than surface details. 
2. Strict Monotonic Information Loss: Once a specific detail is removed at step t, it must never reappear in any later step.
3. Discrete Steps of Abstraction: Each step must be meaningfully more abstract than the previous one, while still retaining the core meaning of the original sentence. Avoid trivial paraphrases that preserve the same level of specificity. Do not just rephrase the same sentence with different words.
4. Core Meaning Preservation: Maintain semantic coherence at every step, Do not introduce new facts or contradic earlier steps.
5. Length and Structure: The number of steps depends on the detail of the sentence (memory item). If the sentence is already abstracted, the list of abstraction will be smaller compared to the one of a very detailed sentence. Do not abstract to the point where the core meaning is lost.

Additionally, the forgetting trajectory must reflect the following domain priors:
Domain: Health
* Forgetting Bias: Exact dosage and measurements, exact dates and specific drug names fade early.
* Retention Bias: Diagnosis, treatment, and outcomes of treatment persist longer.

Domain: Political
* Forgetting Bias: Vote counts, committee names, bill numbers, exact dates, and specific locations fade early.
* Retentaion Bias: Political name, actions, and topic area persist longer.

Domain: Business
* Forgetting Bias: Exact financial figures, specific dates, company names and quarter/year specifics fade early.
* Retention Bias: Executive names, business actions, company category, and directional trend persist longer.

Domain: Sports
* Forgetting Bias: Exact scores, time remaining, stats, venue name, specific player names, weather conditions fade early.
* Retention Bias: Game outcome, team names and type of sport persist longer.

Domain: Social Narrative
* Forgetting Bias: Exact date/time/location fades early
* Retention Bias: Emotional tone persists

Do not return any reasoning or extra phrases or filler sentences. Just return a list that represents the Progressive detail attenuation.

Example 1:
Text(x): I went swimming last Sunday
Progressive detail attenuation(xt): [‘I went swimming last weekend’,‘I did sports last weekend’, ‘I did something last weekend’, ‘I went out’]

Example 2:
Text(x): The patient was prescribed 5mg of DrugX daily for hypertension.
Progressive detail attenuation(xt): [‘The patient was prescribed a medication daily for hypertension.’,
‘The patient was prescribed a medication for hypertension.’, ‘The patient was prescribed something for hypertension.’,
‘The patient was prescribed something for a condition.’, ‘The patient was prescribed something.’]

Example 3:
Text(x): It was cool weather so she went running.
Progressive detail attenuation(xt): [‘Due to weather, she went running’, 'She went running.', 'She exercised.']
"""




## This version assummes the original xt is given, along with x. It can use xt as reference as the original dataset may not be too bad.
## Given the original sentence (memory item) and the human written progressive detail attenuation, your task is to ensure that the provided list is consistent with the following guidelines, if not, rewrite the list to strictly follow the guidelines.

SYSTEM_PROMPT_V2 = """
We are modeling human-like forgetting of a sentence (memory item) by progressively removing episodic and fine-grained details over discrete time steps (t).
Given the original sentence memory item (x) and the human written progressive detail attenuation (xt), your task is to ensure that the provided list is consistent with the following guidelines, if not, rewrite the list to strictly follow the guidelines.

Follow these CORE guidelines strictly:
1. Progressive Detail Attenuation: As t increases, remove fine-grained details first, like numbers, dates, timestamps, dosages, specific locations, specific objects, proper nouns, and precise modifiers. For descriptive modifiers, remove the adjectives modifying objects, and then remove the adjectives of subject and others. Core semantic structure (who did what to whom) should persist longer than surface details. 
2. Strict Monotonic Information Loss: Once a specific detail is removed at step t, it must never reappear in any later step.
3. Discrete Steps of Abstraction: Each step must be meaningfully more abstract than the previous one, while still retaining the core meaning of the original sentence. Avoid trivial paraphrases that preserve the same level of specificity. Do not just rephrase the same sentence with different words.
4. Core Meaning Preservation: Maintain semantic coherence at every step, Do not introduce new facts or contradic earlier steps.
5. Length and Structure: The number of steps depends on the detail of the sentence (memory item). If the sentence is already abstracted, the list of abstraction will be smaller compared to the one of a very detailed sentence. Do not abstract to the point where the core meaning is lost.

Additionally, the forgetting trajectory must reflect the following domain priors:
Domain: Health
* Forgetting Bias: Exact dosage and measurements, exact dates and specific drug names fade early.
* Retention Bias: Diagnosis, treatment, and outcomes of treatment persist longer.

Domain: Political
* Forgetting Bias: Vote counts, committee names, bill numbers, exact dates, and specific locations fade early.
* Retentaion Bias: Political name, actions, and topic area persist longer.

Domain: Business
* Forgetting Bias: Exact financial figures, specific dates, company names and quarter/year specifics fade early.
* Retention Bias: Executive names, business actions, company category, and directional trend persist longer.

Domain: Sports
* Forgetting Bias: Exact scores, time remaining, stats, venue name, specific player names, weather conditions fade early.
* Retention Bias: Game outcome, team names and type of sport persist longer.

Domain: Social Narrative
* Forgetting Bias: Exact date/time/location fades early
* Retention Bias: Emotional tone persists

Do not return any reasoning or extra phrases or filler sentences. Just return a list that represents the Progressive detail attenuation.

Example 1:
Text(x): I went swimming last Sunday
Progressive detail attenuation(xt): [‘I went swimming last weekend’,‘I did sports last weekend’, ‘I did something last weekend’, ‘I went out’]

Example 2:
Text(x): The patient was prescribed 5mg of DrugX daily for hypertension.
Progressive detail attenuation(xt): [‘The patient was prescribed a medication daily for hypertension.’,
‘The patient was prescribed a medication for hypertension.’, ‘The patient was prescribed something for hypertension.’,
‘The patient was prescribed something for a condition.’, ‘The patient was prescribed something.’]

Example 3:
Text(x): It was cool weather so she went running.
Progressive detail attenuation(xt): [‘Due to weather, she went running’, 'She went running.', 'She exercised.']
"""