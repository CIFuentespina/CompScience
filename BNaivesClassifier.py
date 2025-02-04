data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

def prior_probability(data):
    class_counts = {}
    total_instances = len(data)

    for row in data:
        label = row[-1]
        class_counts[label] = class_counts.get(label, 0) + 1

    return {cls: count / total_instances for cls, count in class_counts.items()}, class_counts

def likelihood(data, feature_index, feature_value, class_label):
    count_feature_given_class = sum(1 for row in data if row[feature_index] == feature_value and row[-1] == class_label)
    count_class = sum(1 for row in data if row[-1] == class_label)

    return count_feature_given_class / count_class if count_class else 0

def naive_bayes_predict(data, new_sample):
    priors, class_counts = prior_probability(data)
    classes = class_counts.keys()
    
    probabilities = {cls: priors[cls] for cls in classes}
    
    for cls in classes:
        for i, feature_value in enumerate(new_sample):
            probabilities[cls] *= likelihood(data, i, feature_value, cls)
    
    return max(probabilities, key=probabilities.get)

new_sample = ['Sunny', 'Cool', 'High', 'Strong']
prediction = naive_bayes_predict(data, new_sample)

print(f"Predicted Class for {new_sample}: {prediction}")
