# %%
import os

os.chdir(os.path.dirname(__file__))
# %%
from interview import CustomSLM

# %%
model = CustomSLM("data/1000.txt")
# %%
model.top_k_words("The sky was", 20)
# %%
model.top_n_sentences("The sky was", 3, 4, 5)