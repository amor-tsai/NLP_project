# each module is a file name, if you want to import class from that file, u should use this from module import class-name
from BigramModel import BigramModel
import re
import string


# model = BigramModel("model1","corpus","tx",stopWordList=["the","a"],singlesen=False,smooth=0.2)
model = BigramModel("model1","corpus","tx",stopWordList=[],singlesen=False,smooth=0.2)

print(model.getAll(2))

# if None == re.match(".*[\w]+.*","!"):
#     print(1)
# else:
#     print(2)


