import os
import sys
import src.common.tools as tools
import src.preprocess.dataset as dataset
import src.visualization.visualize as visualization


if "pretrained" in sys.argv[-1].lower():
    pretrained = True if "false" not in [
        argv.lower() for argv in sys.argv
        ] else False
    if len(sys.argv) == 4:
        name = sys.argv[2]
    # Argument is ['-m', 'First or Last Name']
    elif len(sys.argv) == 3:
        name = sys.argv[1]
    elif len(sys.argv) == 2 and sys.argv[0] != "-m":
        name = sys.argv[0]
    else:
        print("Usage: python main.py <First or Last Name> pretrained")
        sys.exit(1)
else:
    pretrained = False
    # Arguments are ['nlp.nlp', '-m', 'First or Last Name']
    if len(sys.argv) == 3:
        name = sys.argv[2]
    # Argument is ['-m', 'First or Last Name']
    elif len(sys.argv) == 2:
        name = sys.argv[1]
    elif len(sys.argv) == 1 and sys.argv[0] != "-m":
        name = sys.argv[0]
    else:
        print("Usage: python main.py <First or Last Name>")
        sys.exit(1)
# print(text)
# sys.stdout.flush()
visualization.visualize(classes=dataset.alldata.labels_uniq, 
                        pretrained=pretrained, 
                        name=name) if "api" not in os.getcwd() else None
result = tools.model.predict(name, pretrained)

if "api" not in os.getcwd():
    print(f"The name {name} is most likely a {str(result).split(',')[0][2:-1]} name.") 
else:
    print(result)
    sys.stdout.flush()
