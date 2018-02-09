import sklearn.metrics
import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np
import scipy

def plot_roc(actual, pred, num_classes=None, class_names=None,
             title='Receiver operating characteristic',
             #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            ):
    if not num_classes:
        num_classes = actual.shape[1]
    if not class_names:
        class_names = ['Class [%d]'%j for j in range(num_classes)]
    
    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], thres[i] = sklearn.metrics.roc_curve(actual[:, i], pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(actual.ravel(), pred.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(dpi=300)
    h = plt.plot(fpr['micro'], tpr['micro'], label='micro-average (AUC = %0.2f)'%(roc_auc['micro']), linestyle=':')
    plt.plot(fpr['macro'], tpr['macro'], label='macro-average (AUC = %0.2f)'%(roc_auc['macro']), linestyle=':')
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='%s (AUC = %0.2f)'%(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.gca().set_aspect(1)
    plt.legend(loc="lower left", bbox_to_anchor=(1.04,0))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
    
    return fpr, tpr, roc_auc, thres
    
def diag_thres(fpr, tpr, thres):
    best_thres = []
    best_j = []

    A = np.array([[1, -1],
                  [-1, 1]])
    A = A / np.linalg.norm(A)
    
    for x, y, t in zip(fpr.values(), tpr.values(), thres.values()):
        y = np.stack((x,y),axis=0) - [[1,],[0,]]
        x, _, _, _ = np.linalg.lstsq(A, y)
        d = np.linalg.norm(y-x, axis=0)
        j = np.argmin(d)
        best_thres.append(t[j])
        best_j.append(j)

    return best_thres, best_j

def best_thres(actual, pred):
    sorted_pred = np.array(pred, copy=True)
    sorted_pred.sort()
    accs = []
    for thres in sorted_pred:
        pred_class = pred > thres
        acc = sklearn.metrics.accuracy_score(actual, pred_class)
        accs.append(acc)    
    best_j = np.argmax(accs)
    best_thres = sorted_pred[best_j]
    
    return best_thres, best_j

def plot_confusion_matrix(cm, normalize=False,
                          num_classes=None, class_names=None,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not num_classes:
        num_classes = cm.shape[0]
    if not class_names:
        class_names = ['Class [%d]'%j for j in range(num_classes)]
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    #print(cm)
    plt.figure(dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.matshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def path_image_loader(path):
    image = cv2.imread(path)
    return image

def inception_postprocessing(image):
    return (np.array(image)+1)/2

def concat_plot(shape, loc, span, samples, fig=None):
    ax = plt.subplot2grid(shape, loc, rowspan=span[0], colspan=span[1], fig=fig)
    h = ax.imshow(np.concatenate(samples, axis=1))
    #plt.axis('off')
    ax.set_axis_off()
    #h.axes.get_xaxis().set_visible(False)
    #h.axes.get_yaxis().set_visible(False)
    return h

def error_report(actual, pred, num_classes=None, class_names=None, top_n=None, top_m=None, samples=None, loader=path_image_loader, postprocessing=None, plot_samples=concat_plot):
    actual_n = actual.argmax(axis=1)
    pred_n = pred.argmax(axis=1)
    if not num_classes:
        num_classes = max(actual_n.max(), pred_n.max()) + 1
    if not class_names:
        class_names = ['Class [%d]'%j for j in range(num_classes)]
    if not top_n:
        top_n = num_classes * num_classes
    if not top_m:
        top_m = num_classes * num_classes
        
    confmat = sklearn.metrics.confusion_matrix(actual_n, pred_n)
    confmat -= np.diag(np.diag(confmat))
    confmat_idx = np.argsort(-confmat, axis=None) # sort descending - index 
    confmat_idx = confmat_idx[:(confmat>0).sum()] # remove correct classes
    confmat_idx = list(zip((confmat_idx // num_classes), (confmat_idx % num_classes)))
    
    confmatlist = {}
    for i, (a_n, p_n) in enumerate(zip(actual_n, pred_n)):
        if (a_n, p_n) in confmatlist:
            confmatlist[(a_n, p_n)].append(i)
        else:
            confmatlist[(a_n, p_n)] = [i]

    for a, p in confmat_idx[:top_n]:
        print('%s -> %s: %d' % (class_names[a], class_names[p], confmat[a,p]))
        sorted_list = sorted( confmatlist[(a,p)],  key=lambda i:pred[i,a] )
        
        print('Error samples:', sorted_list)
        
        j = 0
        if samples:
            error_samples = []
        plt.subplots(dpi=300,figsize=(20, 8))
        for i in sorted_list:
            #print np.stack((actual[i],pred[i]))
            ind = np.arange(num_classes)  # the x locations for the groups
            width = 0.35       # the width of the bars
            #ax = plt.subplot(2, top_m, j+1)
            ax = plt.subplot2grid((3, top_m), (0, j))
            ax.bar(ind, actual[i]+0.05, width, -0.05, color='g')
            ax.bar(ind + width, pred[i]+0.05, width, -0.05, color='r')
            ax.set_ylim((-0.05,1))
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(class_names, rotation=-30) #rotation_mode='anchor', ha='left'
            import matplotlib.ticker
            majorLocator = matplotlib.ticker.MultipleLocator(1)
            ax.yaxis.set_major_locator(majorLocator)
            plt.title(str(i))
            plt.grid(True, which='major', axis='y')

            if samples:
                sample = loader(samples[i])
                if callable(postprocessing):
                    sample = postprocessing(sample)
                error_samples.append(sample)
            j += 1
            if j >= top_m:
                break
        
        if samples:
            h = plot_samples((3, top_m), (1, 0), (2, top_m), error_samples)
        plt.show()
