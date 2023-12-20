import matplotlib.pyplot as plt
import numpy as np

# Data for the chart
labels = ['First try', '', 'SVM Data enhanced', '', '', '', 'CamenBERT', 'Pre-trained', 'SVM',
          'test-size = 1%', 'New features', 'test/train size', '1st NN', '2nd NN', '3rd NN', 'SVM Cross-validation', '']
y_values = [0.317, 0.323, 0.556, 0.529, 0.558, 0.53, 0.565, 0.543, 0.489, 0.586, 0.591, 0.595, 0.172, 0.56, 0.565, 0.597, 0.597]

# Filter out empty strings from labels
filtered_labels = [label if label != '' else ' ' for label in labels]

# Create a line chart
x_values = np.arange(len(labels))
plt.plot(x_values, y_values, marker='o')

# Add title and labels
plt.title('Score Evolution')
plt.xlabel('X-axis Title')
plt.ylabel('Score')

# Set x-axis ticks and labels
plt.xticks(x_values, filtered_labels, rotation=45, ha='right')

# Show the chart
plt.show()

# Save the chart as a PNG file
plt.savefig('chart.png')

# Show the chart
plt.show()
