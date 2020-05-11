% STEP #1

% Enable plots
plots = false;

% The number of patterns in a category
num_patterns = 1584;

% The category separation point
separation = 1585;

% The ratio between train and test patterns
tt_ratio = 0.8;

% The number of train patterns in a category
num_train = floor(num_patterns*tt_ratio);

% The number of test patters in a category
num_test = floor(num_patterns*(1-tt_ratio));

% Load the table
table = readtable('voice.csv');

% Extract the input
input = table2array(table(:,1:end-1));

% Extract the label
label = categorical(table2array(table(:,end)));

% Construct the train dataset
train_input = [input(1:num_train,:); input(separation:separation+num_train-1,:)];
train_label = [label(1:num_train,:); label(separation:separation+num_train-1,:)];

% Construct the test dataset
test_input = [input(num_train+1:num_train+num_test+1,:); input(separation+num_train:separation+num_train+num_test,:)];
test_label = [label(num_train+1:num_train+num_test+1,:); label(separation+num_train:separation+num_train+num_test,:)];

% STEP #2

% Compute means and standard deviations
means = mean(train_input);
stds = std(train_input);

% Compute mean errors
mean_errors = 1.96*stds;

% Compute medium, skewness and kurtosis values
medn = median(train_input);
skew = skewness(train_input);
kurt = kurtosis(train_input);

% Compute medium, skewness and kurtosis errors
medn_err = 0.01*num_train;
skew_err = sqrt(6*num_train*(num_train-1)/(num_train-2)/(num_train+1)/(num_train+3));
kurt_err = sqrt(24*num_train*((num_train-1)^2)/(num_train-3)/(num_train-2)/(num_train+3)/(num_train+5));

% Plot box plots for similar features
if plots==true
    figure
    boxplot([train_input(:,1) train_input(:,12)]);
    figure
    boxplot([train_input(:,18) train_input(:,19)]);
end

% STEP #3

% Perform one-sample Kolmogorov-Smirnov test
normality = zeros(2, 20);
for i=1:20
    [h,p] = kstest(train_input(:,i));
    normality(1,i) = h;
    normality(2,i) = p;
end

% STEP #5

% Analyze correlations between features
if plots==true
    figure;
    corrplot(table(:,1:20));
end

% STEP #6

% Exclude fully correlated features from the train and the test datasets
train_input_2 = [train_input(:,1:11) train_input(:,13:18) train_input(:,20)];
test_input_2 = [test_input(:,1:11) test_input(:,13:18) test_input(:,20)];

% Perform PCA and extract factors
[coeff,score,latent] = pca(train_input_2);
train_input_3 = train_input_2 * coeff;
test_input_3 = test_input_2 * coeff;

% Analyze correlations between factors
if plots==true
    figure;
    corrplot(train_input_3);
end

% Exclude fully correlated factors from the train and the test datasets
train_input_3 = train_input_3(:,1:17);
test_input_3 = test_input_3(:,1:17);

% STEP #6.1

% Prepare the train dataset
train = {
    train_input, train_input_2, train_input_3
};

% Prepare the test dataset
test = {
    test_input, test_input_2, test_input_3
};

% STEP #7 (naive Bayes classifier)

% Train the model
bayes = cell(1,3);
for i=1:3
    bayes{i} = fitcnb(train{i},train_label);
end

% Plot 
roc_labels = {
    'ROC (original DS)', 'ROC (reducted DS)', 'ROC (reducted DS after PCA)'
};
if plots==true
    for i=1:3
        [~,scores] = resubPredict(bayes{i});
        [X,Y,T,AUC] = perfcurve(train_label,scores(:,2),'male');
        figure;
        plot(X,Y);
        xlabel('FPR');
        ylabel('TPR');
        title(roc_labels{i});
    end
end

% Assess the model
bayes_metrics = zeros(3,3);
for i=1:3
    prediction = predict(bayes{i},test{i});
    cm = confusionmat(test_label,prediction);
    tp = cm(1,1);
    tn = cm(2,2);
    fp = cm(1,2);
    fn = cm(2,1);
    bayes_metrics(i,1) = (tp + tn) / (tp + tn + fp + fn);
    bayes_metrics(i,2) = tp / (tp + fp);
    bayes_metrics(i,3) = tp / (tp + fn);
end

% STEP #8 (logistic regression)

% Train the model
glm = cell(1,3);
for i=1:3
    glm{i} = glmfit(train{i},train_label,'binomial');
end

% Assess the model
glm_metrics = zeros(3,3);
for i=1:3
    z = glm{i}(1) + test{i} * glm{i}(2:end);
    z = 1 ./ (1 + exp(-z));
    prediction = categorical(discretize(z, 2, 'categorical', {'female', 'male'}));
    cm = confusionmat(test_label,prediction);
    tp = cm(1,1);
    tn = cm(2,2);
    fp = cm(1,2);
    fn = cm(2,1);
    glm_metrics(i,1) = (tp + tn) / (tp + tn + fp + fn);
    glm_metrics(i,2) = tp / (tp + fp);
    glm_metrics(i,3) = tp / (tp + fn);
end

% STEP #9 (SVM)

% Train the model
svm = cell(1,3);
for i=1:3
    if i==3
        svm{i} = fitcsvm(train{i},train_label,'Standardize',true);
    else
        svm{i} = fitcsvm(train{i},train_label);
    end
end

% Assess the model
svm_metrics = zeros(3,3);
for i=1:3
    prediction = predict(svm{i},test{i});
    cm = confusionmat(test_label,prediction);
    tp = cm(1,1);
    tn = cm(2,2);
    fp = cm(1,2);
    fn = cm(2,1);
    svm_metrics(i,1) = (tp + tn) / (tp + tn + fp + fn);
    svm_metrics(i,2) = tp / (tp + fp);
    svm_metrics(i,3) = tp / (tp + fn);
end
