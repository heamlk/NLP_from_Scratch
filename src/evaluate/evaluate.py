# ================================================
# Evaluating the Recurrent Neural Network (RNN)
# ================================================
import torch


import src.evaluate.evaluate as evaluate
import src.common.tools as tools
import src.model.train as train
import src.preprocess.dataset as dataset

def evaluate(rnn, testing_data, classes, pretrained=True, name="Preston"):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            if pretrained:
                guess, guess_i = tools.model.label_from_output(output, classes, name)
            else:
                guess, guess_i = rnn.label_from_output(output, classes, name)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing each row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    return confusion

def confusion_matrix(pretrained=False, name="Preston"):
    confusion = evaluate(tools.model if pretrained else train.rnn_model, dataset.test_set,
                        classes=dataset.alldata.labels_uniq, pretrained=pretrained, name=name)
    return confusion
