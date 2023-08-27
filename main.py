from linreg import *
import sys, getopt
import os

def main(argv):
    trainfile = ""
    valfile = ""
    testfile = ""
    outpath = ""
    section = 0
    opts, args = getopt.getopt(argv,"hi:o:",["train_path=","val_path=","test_path=","out_path=","section="])
    for opt, arg in opts:
        if (opt == "--train_path"):
            trainfile = arg
        elif (opt == "--val_path"):
            valfile = arg
        elif (opt == "--test_path"):
            testfile = arg
        elif (opt == "--out_path"):
            outpath = arg
        elif (opt == "--section"):
            section = int(arg)

    X_train, Y_train = read_train_data(trainfile)
    X_val, Y_val = read_train_data(valfile)
    X_test,samples = read_test_data(testfile)
    
    outfile =os.path.join(outpath, 'out.csv')
    f = open(outfile, 'w')
    # print(samples)

    # print ('Train file is ', trainfile)
    # print ('Val file is ', valfile)
    # print ('Test file is ', testfile)
    # print ('Out file is ', outfile)
    # print ('Section ', section)

    if (section == 1):
        train_maxit = 10000
        train_reltol = 0.0001
        train_mode = 1
        train_learning_rate = 0.001

        w_train = gradient_descent(X_train, Y_train, mode = train_mode, learning_rate= train_learning_rate, 
                rel_thresh = train_reltol,  maxit= train_maxit, Xv=X_val, Yv = Y_val)

        Y_pred = X_test.dot(w_train)

        print("----------------------------------------------------------------------------")
        print("TRAIN ERROR:")
        print("Train MSE:",mse_gd(w_train, X_train, Y_train))
        print("Train MAE:",mae_gd(w_train, X_train, Y_train))
        print("-------------------------------")
        print("VALIDATION ERROR:")
        print("Validation MSE:", mse_gd(w_train, X_val, Y_val))
        print("Validation MAE:", mae_gd(w_train, X_val, Y_val))
        print("-------------------------------")

        for i in range(len(Y_pred)):
            entry = samples[i] + "," + str(Y_pred[i])
            print(entry, file = f)

    elif (section == 2):
        train_maxit = 10000
        train_lmda = 5
        train_mode = 1
        train_reltol = 0.00001
        train_learning_rate = 0.001
        
        w_train = gradient_descent_ridge(X_train, Y_train, mode = train_mode, learning_rate= train_learning_rate, 
                        rel_thresh = train_reltol, lmda = train_lmda, maxit= train_maxit, Xv=X_val, Yv = Y_val)
        
        Y_pred = X_test.dot(w_train)

        print("----------------------------------------------------------------------------")
        print("TRAIN ERROR:")
        print("Train MSE:",mse_gd(w_train, X_train, Y_train))
        print("Train MAE:",mae_gd(w_train, X_train, Y_train))
        print("-------------------------------")
        print("VALIDATION ERROR:")
        print("Validation MSE:", mse_gd(w_train, X_val, Y_val))
        print("Validation MAE:", mae_gd(w_train, X_val, Y_val))
        print("-------------------------------")


        for i in range(len(Y_pred)):
            entry = samples[i] + "," + str(Y_pred[i])
            print(entry, file = f)

    elif (section == 5):
        w_train = logistic_regression(X_train, Y_train, learning_rate= 0.001, maxit=500, Xv = X_val, Yv = Y_val)

        print("TRAIN PREDICTIONS")
        correct = 0
        total = 0
        for j in range(len(X_train)):
            pred = logistic_test(w_train, X_train[j])
            actual = Y_train[j]
            if (pred == actual):
                correct += 1
            total += 1
            # print(logistic_test(w_train, X_train[j]), Y_train[j])
        acc = correct/total
        print("TRAIN ACCURACY:", acc)
        print("---------------------------------------------------------")
        print("VALIDATION PREDICTIONS")
        correct = 0
        total = 0
        for j in range(len(X_val)):
            pred = logistic_test(w_train, X_val[j])
            actual = Y_val[j]
            if (pred == actual):
                correct += 1
            total += 1
            # print(logistic_test(w_train, X_train[j]), Y_train[j])
        acc = correct/total
        print("VALIDATION ACCURACY:", acc)

        # TRAINING
        for j in range(len(X_test)):
            pred = logistic_test(w_train, X_test[j])
            entry = samples[j] + "," + str(pred)
            print(entry, file = f)
    
    elif (section == 8):
        w_train = logistic_one_vs_all(X_train, Y_train, learning_rate= 0.001, maxit=800, Xv = X_val, Yv = Y_val)
        print("TRAIN PREDICTIONS")
        correct = 0
        total = 0
        for j in range(len(X_train)):
            pred = logistic_test(w_train, X_train[j])
            actual = Y_train[j]
            if (pred == actual):
                correct += 1
            total += 1
            # print(logistic_test(w_train, X_train[j]), Y_train[j])
        acc = correct/total
        print("TRAIN ACCURACY:", acc)
        print("---------------------------------------------------------")
        print("VALIDATION PREDICTIONS")
        correct = 0
        total = 0
        for j in range(len(X_val)):
            pred = logistic_test_one_vs_all(w_train, X_val[j])
            actual = Y_val[j]
            if (pred == actual):
                correct += 1
            total += 1
            # print(logistic_test(w_train, X_train[j]), Y_train[j])
        acc = correct/total
        print("VALIDATION ACCURACY:", acc)

        # TRAINING
        for j in range(len(X_test)):
            pred = logistic_test_one_vs_all(w_train, X_test[j])
            entry = samples[j] + "," + str(pred)
            print(entry, file = f)

    else:
        print("Unknown section", section, "must be either 1, 2, 5 or 8!")

    f.close()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])  