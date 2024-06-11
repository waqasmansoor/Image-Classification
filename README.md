<h2>
  Image Classification with CNN and Supervised and Semi-Supervised Decision Trees
<h2>

<h3>
  How to Run Code
</h3>
<p>
  ```
  pip install -r requirements.txt
  ```
<p>

<h3>
Model Training
</h3>

<p>
  Run Decision Tree Training
  ```
    python main.py --model sup
  ```
</p>

<p>
 Run Decision Tree with Principle Component Analysis
 ```
  python main.py --model sup --pca
 ```
</p>

<p>
Run Semi-Supervised Training with Decision Tree
```
python main.py --model ss
```
</p>

<p>
Run Semi-Supervised Training with Decision Tree and PCA
```
python main.py --model ss --pca
```
</p>
