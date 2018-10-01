
from  out_predict import workflow, workdata as wk, ds

ds.dataset('https://github.com/jcirrone/GRN/data')
ds.download_preprocess(in_dir='grn/my_dir/')
Xtrain, Ytrain, Xval, Yval, Xtest, _ = ds.split_train_val_test(split=[.8, .1], shuffle=True)
wk.set_model('sklearn.RandomForestRegressor', tune=False)
wk.train_and_tune(Xtrain, Ytrain, Xval, Yval)
feat_imps = wk.imp_features()
learnt_params = wk.get_params()
score = wk.score()
Ytest = wk.test(Xtest)
