import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    dataset = pandas.read_csv('databases/dataset_zadanie.csv')
    AXIS_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    dataset = dataset.astype(float)

except:
    input("błąd - nie znaleziono pliku csv\nProgam musi być otwarty w folderze w którym istnieje podfolder 'databases' użyj ENTER by zamknąć")
    exit()


def get_user_input():
    print('Podaj indeksy zmiennych z których chcesz utworzyć wykres: ')
    x = 0
    for name in AXIS_NAMES:
        print(str(x) + '. ' + name)
        x += 1
    run = True
    while run:
        x_input = int(input('Podaj indeks pierwszej zmiennej: '))
        y_input = int(input('Podaj indesx drugiej zmiennej: '))
        if x_input in [0, 1, 2, 3, 4, 5, 6, 7, 8] and y_input in [0, 1, 2, 3, 4, 5, 6, 7, 8]: run = False
        else: print('Podano złe indeksy!\nPodaj jeszcze raz: ')
    user_y = AXIS_NAMES[y_input]
    user_x = AXIS_NAMES[x_input]

    print('Wybrano: ' + user_x + ' oraz ' + user_y)

    return user_x, user_y


def show_correlation_chart(dataset):
    axis_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree function', 'Age', 'Outcome']
    axis_names_short = ['Pregancies', ' Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 'Diabetes\nPedigree\nfunction', 'Age', 'Outcome']
    dataset.corr()
    plt.matshow(dataset.corr())
    plt.xticks(numpy.arange(9), axis_names_short)
    plt.yticks(numpy.arange(9), axis_names)
    plt.show()


def create_scatter():
    user_x, user_y = get_user_input()

    x = dataset[user_x]
    y = dataset[user_y]

    plt.scatter(x, y, s=5)
    plt.title('Współczynnik korelacji: ' + str((x.corr(y)).round(4)))
    plt.xlabel(user_x)
    plt.ylabel(user_y)
    plt.gcf()
    plt.show()


def predict_outcome():
    """funkcja początkowo wybiera 4 elementu mające największy wpływ na zmianną outcome, po czym tworzy model predykcyjny zmiennej outcome"""
    a = dataset.iloc[:, 1:8]
    b = dataset.iloc[:, -1]

    highest_score_elements = SelectKBest(score_func=chi2, k=4)
    fit_elem = highest_score_elements.fit(a, b)

    ds_scores = pandas.DataFrame(fit_elem.scores_)
    ds_columns = pandas.DataFrame(a.columns)

    features_scores= pandas.concat([ds_columns, ds_scores], axis=1)
    features_scores.columns = ['Elements', 'Score']
    result = features_scores.sort_values(by='Score')



    a = dataset[['Age', 'Glucose', 'Insulin']]
    a = a.dropna(axis=0)
    print(a.info())
    b = dataset['Outcome']


    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=100)
    logreg = LogisticRegression()
    logreg.fit(a_train, b_train)
    b_pred=logreg.predict(a_test)
    print (a_test)
    b_pred = b_pred.tolist()
    print((b_pred))

    with open('outcome_prediction.txt', 'w') as file:
        for _ in b_pred:
            file.write(str(int(_)) + '\n')
    file.close()
    print("\nUżywając rozkładu chi^2 możemy zauważyć iż największy wpływ na zmienną outcome mają zmienne 'Age', 'Glucose' i 'Insulin', dlatego" )
    input('Naciśnij ENTER by wrócić do ekranu startowego')


while True:
    print('Aplikacja umożliwia analizę danych statystycznych z pliku csv\nUżytkownik ma możliwość wybrać konkretną funk'
          'cjonalność programu wpisując numer danej funkcji i zatiwerdzająć ENTER\n1. Pokaż wykres korelacji pomiędzy'
          ' zmiennymi\n2. Utwórz wykres kropkowy prezertujący zależności pomiędzy zmiennymi oraz ich korelacje\n3. Uruc'
          'hom model predykcyjny zmiennej outcome\n4. Wyłącz program.')

    user_in = input('Podaj number funkcji którą chcesz wykonać: ')
    if user_in == '1': show_correlation_chart(dataset)
    elif user_in == '2': create_scatter()
    elif user_in == '3': predict_outcome()
    elif user_in == '4': exit()
    print('\n\n')
