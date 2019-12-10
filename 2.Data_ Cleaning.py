# %% [markdown]
# *****************************Data Cleaning*************************

# %% [code]
#fix the outlier where YrSold is earlier than YrBuilt in test set
test.loc[1089]["YrSold"] = 2009
test.loc[1089]["YrActualAge"] = 0

#store the 'Id' column then drop it from original datasets--not useful in reg
#axis = 1 indicates col
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#scatter plot shows distribution of labels and living areas
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# %% [code]
#Delete outliers in the bottom-right corner
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check distribution again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# %% [markdown]
# transform the target variable: saleprice

# %% [code]
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Plot the distribution of salesprice
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# %% [markdown]
# Label is right-skewed. necessary to perform log trans to make it more normally distributed.

# %% [code]
#use the numpy fuction log1p to apply log(1+x): plus 1 to avoid -inf
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# %% [code]
#List the ratio of missing variables
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:20]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(10)

# %% [code]
#concatenate train and test data to clean data together
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all = pd.concat((train, test)).reset_index(drop=True)
all.drop(['SalePrice'], axis=1, inplace=True)
print("all size is : {}".format(all.shape))

# %% [code]
#Assign "None" to missing values accordingly:
for col in ('MSSubClass', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all[col] = all[col].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood in the training set
#median, mean functions are not affected by missing values. first, obtain the median of training data
nbh_lot = train.groupby(train.Neighborhood)[['LotFrontage']].median()
#med_lot = neigh_lot.groupby("Neighborhood")["LotFrontage"].transform("median")
#all["LotFrontage"] = all["LotFrontage"].fillna(med_lot)
#all.loc[all.Neighborhood.isin(neigh_lot.Neighborhood), ['LotFrontage']] = neigh_lot['LotFrontage']
all = all.merge(nbh_lot, on=["Neighborhood"], how='left', suffixes=('','_'))
all['LotFrontage'] = all['LotFrontage'].fillna(all['LotFrontage_']).astype(int)
all = all.drop('LotFrontage_', axis=1)

#all["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
for col in ('MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all[col] = all[col].fillna(0)
#Remove "Utilities"-- For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA .
all = all.drop(['Utilities'], axis=1)
#Functional : data description says NA means typical
all["Functional"] = all["Functional"].fillna("Typ")
#vars with only one NA value, use mode of this var in the training set to prevent data leakage
for col in ('KitchenQual', 'Electrical', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType'):
    all[col] = all[col].fillna(train[col].mode()[0])

#check if there is any remaining missing values
all[all.isna().any(axis=1)]
