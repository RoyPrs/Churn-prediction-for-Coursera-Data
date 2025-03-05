# Statisticsl inference package
from scipy import stats
from scipy.stats import t, chi2, norm, f
import numpy as np
import pandas as pd
from math import sqrt

def f_test_variance(df, column, target, significance=0.95):
    sample1 = df[df[target] == True]
    sample2 = df[df[target] == False]

    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    nu1 = n1 - 1
    nu2 = n2 - 1

    variance1 = np.var(sample1[column], ddof=1, axis=0)
    variance2 = np.var(sample2[column], ddof=1, axis=0)
    
    f_statistics = variance1 / variance2

    left_sig = (1 - significance) / 2
    right_sig = 1 - left_sig
    # print(f"Significance interval = ({left_sig}, {right_sig})")
    
    left_rejection = f.ppf(left_sig, nu1, nu2)
    right_rejection = f.ppf(right_sig, nu1, nu2) # = 1/left_rejection
    
    #print(f"Rejection Region = ({round(left_rejection,2)}, {round(right_rejection,2)})")
    #print(f"f statistics = {round(f_statistics,2)}")
 
    if f_statistics > left_rejection and f_statistics < right_rejection:
        #print("\nPopulation variances are equal.")
        return True
    else:
       # print("\nPopulation variances are not equal.")
        return False


def t_test_mean(df, column, target, significance = 0.05):
    sample1 = df[df[target] == True]
    sample2 = df[df[target] == False]

    equal_variance = f_test_variance(df, column, target, significance)
    ttest = stats.ttest_ind(sample1[column], sample2[column], equal_var=equal_variance)
    #print("ttest output", ttest)
    p_value = ttest.pvalue
    
   # msg_accept = f"""The p-value in this case is {p_value}, which is above the standard thresholds of {significance}. So we don't reject the null hypothesis in favor of the alternative hypothesis and say there is no statistically significant difference in mean {columns} between churned and retained learners."""
    #msg_reject = f"""The p-value in this case is {p_value}, which is below the standard thresholds of {significance}. So we reject the null hypothesis in favor of the alternative hypothesis and say there is a statistically significant difference in mean {columns} between churned and retained learners."""
    return p_value
  #  print(f"Mean of retained: {round(sample1[column].mean(),1)}")
   # print(f"Mean of churned: {round(sample2[column].mean(),1)}")
    #print(f"Mean of all: {round(df[column].mean(),1)}")
    
    #print(f"\nMedian of retained: {round(sample1[column].median(),1)}")
    #print(f"Median of churned: {round(sample2[column].median(),1)}")
    #print(f"Median of all: {round(df[column].median(),1)}")
    
    #print(f"\nSTD of retained: {round(sample1[column].std(),1)}")
    #print(f"STD of churned: {round(sample2[column].std(),1)}")
    #print(f"STD of all: {round(df[columns].std(),1)}")

    #if p_value > significance:
        #return msg_accept
   # else:
       # return msg_reject


def t_test_mean_manual(df, column, target, significance=0.95):
    
    sample1 = df[df[target] == True]
    sample2 = df[df[target] == False]

    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    nu1 = n1 - 1
    nu2 = n2 - 1
    dof = nu1 + nu2
    mu1 = mu2 = 0
    
    variance1 = np.var(sample1[column], ddof=1, axis=0)
    variance2 = np.var(sample2[column], ddof=1, axis=0)
    mean1 = np.mean(sample1[column], axis=0)
    mean2 = np.mean(sample2[column], axis=0)
    print(f"Variance1 = {variance1}, Variance2 = {variance2}")
    print(f"Mean1 = {mean1}, Mean2 = {mean2}")

    left_sig = (1 - significance) / 2
    right_sig = 1 - left_sig
    print(f"Significance Interval = ({left_sig}, {right_sig})")

    variances_equal = f_test_variance(df, column, target, significance=significance)

    if variances_equal:
        pooled_variance = (nu1 * variance1 + nu2 * variance2) / (nu1 + nu2)
        # print(f"Pooled Variance = {pooled_variance}")
        left_rejection = t.ppf(left_sig, df = dof)
        right_rejection = t.ppf(right_sig, df = dof)
        
        t_statistics = ((mean1 - mean2 ) - (mu1 - mu2)) / sqrt(pooled_variance * (1/n1 + 1/n2))
        print(f"t statistics = {t_statistics}")
        print(f"Rejection Criteria: t statistics not in the ({left_rejection}, {right_rejection}) interval")

    else:
        nu = ((variance1/n1 + variance2/n2)**2) / ((variance1 / n1)**2 / nu1 + 
                                                    (variance2/n2)**2 / nu2)
        nu = np.ceil(nu)
        left_rejection = t.ppf(left_sig, df=nu)
        right_rejection = t.ppf(right_sig, df=nu)
       
        t_statistics = ((mean1 - mean2) - (mu1 - mu2)) / sqrt(variance1 / n1 + variance2 / n2)

        print(f"t statistics = {t_statistics}")
        print(f"Rejection Criteria: t statistics not in the ({left_rejection}, {right_rejection}) interval")
		

def z_test_two_proportions(df, column, target, significance=0.95, test=True):
    #Tally the number of successes in each sample
    sample1 = df[df[target] == True]
    sample2 = df[df[target] == False]

    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    x1 = sample1[sample1[column] == True].shape[0]
    x2 = sample2[sample2[column] == True].shape[0]
    print("n1, n2, x1, x2", n1, n2, x1, x2)
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    print(f"p1 = {round(p1,3)}, p2 = {round(p2, 3)}, p = {round(p, 3)}")
   
    z_stat = (p1 - p2) / sqrt(p * (1-p) * (1/n1 + 1/n2))
    
    left_sig = (1 - significance) / 2
    right_sig = 1 - left_sig
    print(f"Significance Interval = ({round(left_sig, 3)}, {round(right_sig, 3)})")
    z_left = norm.ppf(left_sig)
    z_right = norm.ppf(right_sig)
    
    print(f"z statistics = {z_stat}")
    print(f"The rejection criteria: z statistics not in the {round(z_left,2)}, {round(z_right, 2)} interval.")
    
