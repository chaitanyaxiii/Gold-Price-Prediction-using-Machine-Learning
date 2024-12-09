import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Load the dataset
df = pd.read_csv('D:\\Excels\\Gold Prices.csv')

# Preprocess the data
# Convert the 'Month' column to numerical values using Label Encoding
label_encoder = LabelEncoder()
df['Month'] = label_encoder.fit_transform(df['Month'])

# Define the features and target variable
features = ['Month', 'Open', 'High', 'Low', 'Close']
target = 'Close'  # Assuming we are predicting the 'Close' price

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define a function to predict gold price based on user inputs
def predict_gold_price(month, open_price, high_price, low_price, close_price):
    # Convert the month to numerical value using the trained Label Encoder
    month_num = label_encoder.transform([month])[0]
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[month_num, open_price, high_price, low_price, close_price]], 
                              columns=features)
    # Predict the gold price
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Define a function to show the first 20 lines of data
def show_data():
    messagebox.showinfo("First 20 lines of Data", df.head(20).to_string())

# Define a function to show the accuracy
def show_accuracy():
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    precision = mean_squared_error(y_test, y_pred)
    messagebox.showinfo("Accuracy and Precision", f"Accuracy: {accuracy:.2f}\nPrecision (MSE): {precision:.2f}")

# Define a function to show the graph
def show_graph():
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, model.predict(X_test))
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.show()

# Define the function to predict and show the result
def show_result():
    month = month_entry.get()
    open_price = float(open_entry.get())
    high_price = float(high_entry.get())
    low_price = float(low_entry.get())
    close_price = float(close_entry.get())
    predicted_price = predict_gold_price(month, open_price, high_price, low_price, close_price)
    messagebox.showinfo("Prediction Result", f"Predicted gold price: {predicted_price:.2f}")

# Create the main Tkinter window
root = Tk()
root.title("STOCK MARKET PRICE ANALYSIS AND PREDICTION")
root.geometry("800x600")

# Add background image
bg_image = Image.open("C:\\Users\\hp\\Desktop\\Gold_image.jpg")
bg_image = bg_image.resize((800, 600))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Add input labels and entries
input_frame = Frame(root, bg="black")
input_frame.place(relx=0.5, rely=0.3, anchor=CENTER)

Label(input_frame, text="Enter the month:", bg="black", fg="white").grid(row=0, column=0, padx=10, pady=10, sticky=E)
month_entry = Entry(input_frame)
month_entry.grid(row=0, column=1, padx=10, pady=10)

Label(input_frame, text="Enter the open price:", bg="black", fg="white").grid(row=1, column=0, padx=10, pady=10, sticky=E)
open_entry = Entry(input_frame)
open_entry.grid(row=1, column=1, padx=10, pady=10)

Label(input_frame, text="Enter the high price:", bg="black", fg="white").grid(row=2, column=0, padx=10, pady=10, sticky=E)
high_entry = Entry(input_frame)
high_entry.grid(row=2, column=1, padx=10, pady=10)

Label(input_frame, text="Enter the low price:", bg="black", fg="white").grid(row=3, column=0, padx=10, pady=10, sticky=E)
low_entry = Entry(input_frame)
low_entry.grid(row=3, column=1, padx=10, pady=10)

Label(input_frame, text="Enter the close price:", bg="black", fg="white").grid(row=4, column=0, padx=10, pady=10, sticky=E)
close_entry = Entry(input_frame)
close_entry.grid(row=4, column=1, padx=10, pady=10)

# Add buttons
button_frame = Frame(root, bg="black")
button_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

Button(button_frame, text="Show Result", command=show_result, bg="blue", fg="white").grid(row=0, column=0, padx=10, pady=10)
Button(button_frame, text="Show Data", command=show_data, bg="green", fg="white").grid(row=0, column=1, padx=10, pady=10)
Button(button_frame, text="Show Accuracy and Precision", command=show_accuracy, bg="orange", fg="white").grid(row=0, column=2, padx=10, pady=10)
Button(button_frame, text="Show Graph", command=show_graph, bg="purple", fg="white").grid(row=0, column=3, padx=10, pady=10)

# Run the main event loop
root.mainloop()
