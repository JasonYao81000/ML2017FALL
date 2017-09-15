import sys

class wordObj(object):
    def __init__(self, word, id):
        self.word = word
        self.id = id
        self.count = 1

if __name__ == "__main__":
    # Create empty list.
    wordTable = list()

    # Load argument from bash.
    DATA_FILE_PATH = sys.argv[1]

    # Read txt file.
    f = open(DATA_FILE_PATH, 'r')

    # Read line by line.
    for lines in f:
        # Split elements by ' '.
        words = lines.split(' ')

        # If it is end with '\n'.
        if (words[-1][-1] == '\n'):
            # Delete last character.
            words[-1] = words[-1][0:-1]
        
        # Find word in wordTable.
        for word in words:
            # If wordTable is empty.
            if (len(wordTable) == 0):
                # So insert it anyway.
                wordTable.append(wordObj(word, len(wordTable)))
            else:
                # Does it exist in wordtable?
                isExist = False
                # Check each word in wordTable.
                for w in wordTable:
                    # This word exists in wordTable.
                    if (w.word == word):
                        # Increase word count.
                        w.count = w.count + 1
                        # Mark it exists in wordtTable.
                        isExist = True
                        # Exit this for loop to save time.
                        break
                # This word doesn't exist in wordTable.
                if (isExist == False):
                    # So insert it.
                    wordTable.append(wordObj(word, len(wordTable)))

    # Write txt file.
    f = open('Q1.txt', 'w')
    # Write elements line by line.
    for w in wordTable:
        # Write with format: 'word id count'
        f.write('%s %d %d' %(w.word, w.id, w.count))
        # Write new line except last line.
        if (w != wordTable[-1]):
            f.write('\n')