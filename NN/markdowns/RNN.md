### 1. Architecture du RNN

Pour un RNN simple :

$$
h_t = \tanh(b + Wh_{t-1} + UX_t)
$$

$$
y_t = \operatorname{softmax}(Vh_t)
$$

avec :

* $X_t$ : entrée au temps (t)
* $h_t$ : état caché
* $U$ : poids input → hidden
* $W$ : poids récurrent
* $V$ : poids hidden → output
* $y_t$ : probabilités des classes POS
* $\hat y_t$ : véritable classe POS.

---

### 2. Le point important concernant la softmax + cross-entropy

Il faut surtout retenir que :

$$
L_t=-\sum_i \hat y_{t,i}\log(y_{t,i})
$$

avec

$$
y_t=\operatorname{softmax}(z_t),\qquad z_t=Vh_t
$$

donne directement :

$$
\boxed{
\frac{\partial L_t}{\partial z_t}=y_t-\hat y_t
}
$$

**C'est ici que se produit la simplification importante.**

Attention à une correction essentielle par rapport à certaines réponses précédentes : on ne doit pas écrire

$$
\frac{\partial L_t}{\partial y_t}=y_t-\hat y_t.
$$

Cette égalité est fausse.

En réalité :

$$
\boxed{
\frac{\partial L_t}{\partial y_t}
=================================

-\frac{\hat y_t}{y_t}
}
$$

et le Jacobien de la softmax est :

$$
J_{\mathrm{softmax}}
====================

\operatorname{diag}(y_t)-y_ty_t^T.
$$

La règle de la chaîne donne alors :

$$
\frac{\partial L_t}{\partial z_t}
=================================

\frac{\partial L_t}{\partial y_t}
\frac{\partial y_t}{\partial z_t}.
$$

Et **ce produit**, après simplification, donne :

$$
\boxed{
\frac{\partial L_t}{\partial z_t}
=================================
y_t-\hat y_t
}
$$

C'est cette quantité qui intervient ensuite dans le gradient de (V).

---

### 3. Gradient par rapport à (V)

Puisque

$$
z_t=Vh_t,
$$

on obtient :

$$
\boxed{
\frac{\partial L_t}{\partial V}
===============================
(y_t-\hat y_t)h_t^T
}
$$

et sur toute la séquence :

$$
\boxed{
\frac{\partial L}{\partial V}
=============================

\sum_t
(y_t-\hat y_t)h_t^T
}
$$

C'est un **produit extérieur** : si $y_t-\hat y_t$ est de dimension $K$ et $h_t$ de dimension $H$, le gradient de $V$ est $K\times H$.

---

### 4. La dérivée de $\tanh$

Pour:

$$h_t=\tanh(a_t)$$

avec

$$
a_t=b+Wh_{t-1}+UX_t,
$$

on utilise :

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
$$

Comme

$$
h_t=\tanh(a_t),
$$

on peut écrire :

$$
\boxed{
\frac{dh_t}{da_t}=1-h_t^2
}
$$

pour un neurone scalaire.

Pour plusieurs neurones, le (1-h_t^2) doit être compris **élément par élément**.

---

### 5. Le point subtil : (h_{t-1}) dépend bien de (W)

C'est probablement **le point le plus important de notre discussion**.

Lorsque l'on calcule la contribution directe :

$$
\frac{\partial h_t}{\partial W},
$$

on peut temporairement considérer (h_{t-1}) comme constant :

$$
\frac{\partial (Wh_{t-1})}{\partial W}
======================================

h_{t-1}^T.
$$

Cela donne la contribution directe :

$$
(1-h_t^2)h_{t-1}^T.
$$

Mais :

$$
h_{t-1}
=======
\tanh(b+Wh_{t-2}+UX_{t-1})
$$

donc **$(h_{t-1})$ dépend lui-même de $(W)$**.

Il faut donc ajouter la contribution indirecte :

$$
\boxed{
\frac{\partial h_t}{\partial W}
===============================

(1-h_t^2)
\left[
h_{t-1}^T
+
W\frac{\partial h_{t-1}}{\partial W}
\right]
}
$$

C'est précisément cette deuxième partie qui fait apparaître la **récurrence du gradient**.

---

### 6. C'est le principe de la BPTT

La dépendance peut être représentée ainsi :

$$
W
\rightarrow h_{t-2}
\rightarrow h_{t-1}
\rightarrow h_t
\rightarrow y_t
$$

mais également :

$$
W\rightarrow h_t.
$$

Le même poids (W) est donc utilisé à **tous les instants**.

Le gradient par rapport à (W) doit donc additionner toutes les contributions :

$$
\boxed{
\frac{\partial L}{\partial W}
=============================

\sum_t
\frac{\partial L}{\partial W}\bigg|_t
}
$$

C'est ce partage du même (W) à travers le temps qui nécessite la **Backpropagation Through Time (BPTT)**.

---

### 7. La formule conceptuelle à retenir

Pour comprendre la BPTT, retiens surtout cette structure :

$$
\boxed{
\text{gradient actuel}
======================

\text{contribution directe}
+
\text{gradient provenant du futur}
}
$$

ou, schématiquement :

$$
\boxed{
\frac{\partial L}{\partial h_t}
===============================

\underbrace{
\frac{\partial L_t}{\partial h_t}
}*{\text{erreur locale}}
+
\underbrace{
\frac{\partial L}{\partial h*{t+1}}
\frac{\partial h_{t+1}}{\partial h_t}
}_{\text{erreur venant du futur}}
}
$$

C'est **la formule conceptuelle fondamentale de la BPTT**.

En anglais :

> **The gradient propagates backward through time. This is the fundamental principle of BPTT.**

Enfin, pour ton exemple de POS tagging, la chaîne complète à garder en tête est :

$$
X_t
\rightarrow
h_t
\rightarrow
y_t
\rightarrow
L_t
$$

et, lors de la rétropropagation :

$$
L_t
\rightarrow
y_t
\rightarrow
h_t
\rightarrow
h_{t-1}
\rightarrow
h_{t-2}
\rightarrow\cdots
$$

**C'est cette remontée dans le temps qui constitue la BPTT.**
