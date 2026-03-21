

# forward index maps location to value,
# an inverted index maps value to location
# e.g. token 'matrix' -> [1, 5, 10] movies ids
class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        pass

