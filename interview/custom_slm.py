from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .utils import TokeniserOutput


class CustomSLM:
    """A custom small language model that uses a restricted vocabulary set to generate text and analyse probabilities."""

    def __init__(self, vocabulary_file: str):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
        self.model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-medium"
        )
        self.model.eval()  # set to evaluation mode
        self.vocabulary_mask = self._load_vocabulary_mask(vocabulary_file)
        self.exclude_vocabulary_mask = torch.logical_not(self.vocabulary_mask)

    def _load_vocabulary_mask(self, filepath: str) -> torch.Tensor:
        """Loads a list of allowed vocabulary from a file on disk
        and computes the vocabulary mask needed to ensure correct
        probabilities and completions.
        
        The vocabulary mask is a boolean vector with the same size 
        as the tokeniser `vocab_size`.
        
        This mask should allow all tokens resulting from tokenising
        the words listed in the supplied file, along with some
        punctuation given in the list [",", ".", "'", "!", "?"].
        
        Note: Words with spaces in front may be tokenised differently
        to words without spaces. Ensure that the mask also allows
        words with spaces in front of them.

        Args:
            filepath (str): A file containing the allowed vocabulary for this model, separated on new lines.

        Returns:
            torch.Tensor: A boolean mask tensor which can be used to select only the allowed tokens in the supplied file.
        """
        with open(filepath, 'r') as f:
            vocabulary=f.readlines()
        
        vocab_size = self.tokenizer.vocab_size
        vocab_mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        for word in vocabulary:
            tokens = self.tokenise(word)
            token_ids = tokens['input_ids'][0]
            vocab_mask[token_ids] = True
            
            tokens = self.tokenise(' '+word)
            token_ids = tokens['input_ids'][0]
            vocab_mask[token_ids] = True
            
        
        # Additional punctuations
        for punctuation in [",", ".", "'", "!", "?"]:
            token_index = self.tokenizer(punctuation)["input_ids"][0]
            vocab_mask[token_index] = True
            
        return vocab_mask

    def tokenise(self, text: str) -> TokeniserOutput:
        """Encodes some text into tokens, which act as an embedding for the train model.

        Args:
            text (str): The input sentence to tokenise.

        Returns:
            TokeniserOutput: A dictionary containing an attention mask and the tokens themselves.
        """
        return self.tokenizer(text, return_tensors="pt")

    def raw_next_token_logits(self, tokenisation: TokeniserOutput) -> torch.Tensor:
        """Gets the raw logits that relate to the probabilities of the next token, unadjusted for the current vocabulary.

        Args:
            encoded_text: The tokenised and encoded text, returned from the `tokenise` method.

        Returns:
            torch.Tensor: A vector with self.model. elements, representing the raw logit outputs of the next token.
        """
        return torch.flatten(self.model(**tokenisation).logits[0, -1, :])

    def next_token_probabilities(self, tokenisation: TokeniserOutput) -> torch.Tensor:
        """Get the probabilities of the immediate next token, as given by the model and
        re-weighted according to the vocabulary mask, such as all non-vocabulary tokens
        have a probability of 0.

        Args:
            tokenisation (TokeniserOutput): The tokenised text, as output by `tokenise(text)`.

        Returns:
            torch.Tensor: A vector containing the normalised probability of each token, with the same size as the original vocabulary.
        """
        logits = self.raw_next_token_logits(tokenisation)
        logits[self.exclude_vocabulary_mask] = float('-inf')
        # logits = logits * self.vocabulary_mask
        softmax_logits = torch.softmax(logits, dim=0)

        return softmax_logits


    def top_k_tokens(self, tokenisation: TokeniserOutput, k: int) -> torch.Tensor:
        """Return the top `k` next tokens from the initial text.

        Args:
            text (str): The untokenised text.
            k (int): The number of tokens to select, must be >= 0.
        

        Returns:
            torch.Tensor: The next tokens, sorted from highest probability to lowest.
        """
        valid_logits = self.next_token_probabilities(tokenisation)
        sorted_indices = torch.argsort(valid_logits, descending=True)
        top_k_indices = sorted_indices[:k]
        # print('top k indices',top_k_indices)
        return top_k_indices
        # pass
    
    def top_k_words(self, sentence: str, k: int) -> List[str]:
        """Return the top `k` next decoded tokens from the initial text.

        Args:
            text (str): The untokenised text.
            k (int): The number of tokens to select, must be >= 0.

        Returns:
            List[str]: The next tokens, decoded into words, sorted from highest probability to lowest.
        """
        tokens = self.tokenise(sentence)
        top_tokens = self.top_k_tokens(tokens, k)
        return [self.tokenizer.decode(t) for t in top_tokens]

    def tokens_log_probability(self, tokenisation: TokeniserOutput) -> float:
        """Get the joint log probability of a sequence of tokens, according
        to the re-weighted (and normalised) token probability distribution
        represented by the logits of the masked model.
        
        This method only uses a single pass through the model to access the logit
        probabilities.
        
        The probability of the first token is taken to be equal to 1, and is
        not output by the model.
        
        Args:
            tokenisation (TokeniserOutput): The tokenised text.

        Returns:
            float: The log of the joint probability of all tokens in the text. 
        """
        logits = self.raw_next_token_logits(tokenisation)
        logits[self.exclude_vocabulary_mask] = float('-inf')
        probabilities = torch.softmax(logits, dim=-1)
        probabilities[0] = 1.
        probabilities = probabilities[probabilities!=0]
        log_probabilities = torch.log(probabilities)
        return log_probabilities.sum().item()
    

    def text_log_probability(self, text: str) -> float:
        """Get the joint log probability of an entire piece of text, according
        to the re-weighted (and normalised) token probability distribution.
        
        Args:
            text (str): The untokenised text.

        Returns:
            float: The log of the joint probability of all tokens in the text. The probability of the first token is taken to be equal to 1.
        """
        tokenised_text = self.tokenise(text)
        # print('tokenised text:', tokenised_text)
        return self.tokens_log_probability(tokenised_text)

    def top_n_sentences(self, text: str, n: int, num_tokens: int, k: int) -> List[str]:
        """Return a ranked list of the highest probability sentence completions
        with n remaining tokens, where the last token is forced to be a full stop.
        
        The list of candidates is constructed by recursively searching for the next
        token, searching at most the top `k` tokens for each next token.
        
        Example: top_n_sentences("The capital", 2, 4, 2)
        
        The top two next tokens are "city" and "of".
        
        For "The capital of", the next top two tokens are "the" and "a". 
        
        For both "The capital of the" and "The capital city of a", the next top
        tokens are "country" and "state". The final token is forced to be a full
        stop ("."). These candidates are added to the pool, along with those
        constructed by using "city" instead of "of" as the first additional token.
        
        In total, the candidate pool is therefore:
        - "The capital city of the."
        - "The capital city of a."
        - "The capital city is a."
        - "The capital city is the."
        - "The capital of the country."
        - "The capital of the state."
        - "The capital of a country."
        - "The capital of a state."
        
        Of these candidates, the two with the highest probability are
        - "The capital of the country."
        - "The capital of the state."
        
        Which are returned in the list of strings, ordered from highest to lowest probability.

        Args:
            text (str): The untokenised partial text to be completed.
            n (int): The maximum number of sentences to return.
            num_tokens (int): The number of remaining tokens, including the full stop, which are to be added to the current sentence.
            k (int): The breadth of the tree search - i.e. the number of tokens checked at each token length and the number of candidates.

        Returns:
            List[str]: A list of full completions of the text, ranked in order of probability from highest to lowest. Has between 1 and n inclusive elements.
        """
        # pass
                    
    

