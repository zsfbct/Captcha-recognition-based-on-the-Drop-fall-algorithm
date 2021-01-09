'''
对测试集给出预测结果和模型评估结果
'''

import numpy as np
from chars import find_chars,find_chars_water_extract
from char_classifier import CharClassifier

class OCRModel:
    def __init__(self):
        self.char_classifier = CharClassifier()

        self.char_classifier.load_weights()

        self.char_classifier.summary()

    def predict(self, X):
        num_classes, text_size = 46, 4
        char_size = self.char_classifier.layers[0].input_shape[1:3]

        y0 = np.zeros([X.shape[0], text_size, num_classes]).astype(np.float32)
        y1= np.zeros([X.shape[0], text_size, num_classes]).astype(np.float32)

        for i in range(0, X.shape[0]):
            chars0 = find_chars(X[i, :, :, 0], (60,40),4)
            chars1=find_chars_water_extract(X[i, :, :, 0], (60,40),4)
            y0[i] = self.char_classifier.predict(chars0)
            y1[i] = self.char_classifier.predict(chars1)

        return (y0,y1)

    def predict_labels(self, X):
        y=np.zeros([X.shape[0], 4]).astype(np.uint8)
        (y0,y1)=self.predict(X)

        # # 取两个概率最大的一个
        for i in range(X.shape[0]):
            for j in range(4):
                if y0[i,j].max() <=y1[i,j].max():
                    y[i,j]=y1[i,j].argmax()
                else:
                    y[i, j] = y0[i,j].argmax()
        return y

    def labels_to_text(self, y_labels):
        '''
        将4*46的0-1标签转变为字符对应的数字
        '''
        def to_char(label):
            data = np.load('./preprocessed-data-test.npz', allow_pickle=True)
            alphabet= data['alphabet']
            if label < 0 or label >= len(alphabet):
                return ' '
            return alphabet[label]
        return np.vectorize(to_char, ['S1'])(y_labels)

    def predict_text(self, X):
        '''
        给出四字结果预测
        '''
        return self.labels_to_text(self.predict_labels(X))



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from metrics import summary

    dataset = np.load('./preprocessed-data-test.npz')
    X, y = dataset['X'], dataset['y']

    model = OCRModel()

    # 展示部分预测结果图片
    indices = np.random.choice(np.arange(0,X.shape[0]), size=12)
    X_batch=[]
    y_batch=[]
    for i in indices:
        X_batch.append(X[i])
        y_batch.append(y[i])
    X_batch=np.array(X_batch)
    y_batch =np.array(y_batch)


    texts = [''.join([char.item().decode() for char in text]) for text in OCRModel().labels_to_text(y_batch.argmax(axis=2))]
    texts_pred = [''.join([char.item().decode() for char in text]) for text in model.predict_text(X_batch)]


    rows, cols = X_batch.shape[0] // 3, 3

    fig, ax = plt.subplots(rows, cols, figsize=(25, 15))
    for i in range(0, rows):
        for j in range(0, cols):
            k = i * cols + j
            plt.sca(ax[i, j])

            plt.imshow(X_batch[k, :, :, 0], cmap='gray')
            plt.title('Labels: "{}", Prediction: "{}"'.format(texts[k], texts_pred[k]))

            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()

    # 对测试集进行预测并计算模型准确率
    X_test, y_test =X,y

    print('Predicting captcha text images...')
    print('Number of images: {}'.format(X_test.shape[0]))
    y_labels = y.argmax(axis=2)
    y_pred_labels = model.predict_labels(X)


    print('Evaluation summary:')
    summary(y_labels, y_pred_labels)

    plt.show()
