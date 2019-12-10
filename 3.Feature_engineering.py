# %% [markdown]
# ******************************Feture Engineering******************************
# transform numerical variables that are essentially strings: ints -> categories
# 
# This also prepares for label encoding step next

# %% [code]
#Changing OverallCond into a categorical variable
all['OverallCond'] = all['OverallCond'].apply(str)

#Year and month sold are transformed into categorical features.
all['YrSold'] = all['YrSold'].astype(str)
all['MoSold'] = all['MoSold'].astype(str)

# %% [code]
#LabelEncode categorical variables with orders: different numbers in the same col
cates = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cates:
    lbl = LabelEncoder() 
    lbl.fit(list(all[c].values)) 
    all[c] = lbl.transform(list(all[c].values))

#add a feature: total areas
all['TotalSF'] = all['TotalBsmtSF'] + all['1stFlrSF'] + all['2ndFlrSF']

# %% [code]
#use one hot code transfer categorical values -- preparing for PCA
all = pd.get_dummies(all)
print(all.shape)

# %% [markdown]
# Skewed features

# %% [code]
num_vars = all.dtypes[all.dtypes != "object"].index

# Check the skew of all numerical features
skewed_vars = all[num_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkewness of numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_vars})
skewness.head(15)

# %% [markdown]
# Box Cox Transformation of (highly) skewed features

# %% [code]
skewness = skewness[abs(skewness) > 0.8]
print("There are {} significantly skewed numerical features".format(skewness.shape[0]))

skewed_features = skewness.index
lam = 0.25
for f in skewed_features:
    all[f] = boxcox1p(all[f], lam)


# %% [code]
#Getting the new train and test sets.
train = all[:ntrain]
test = all[ntrain:]

# %% [code]
#standardize dataset -- preparing for PCA
scaler = StandardScaler()
#fit on training set only.
scaler.fit(train)
# Apply transform to both the training set and the test set.
train = scaler.transform(train)
test = scaler.transform(test)

# %% [code]
#pca: reduction dimensionality of features
#make an instance of the Model: scikit-learn choose the minimum number of principal components such that 95% of the variance is retained
pca = PCA(.95)
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)

