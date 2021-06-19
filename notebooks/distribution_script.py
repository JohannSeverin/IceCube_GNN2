# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import sqlite3, os, sys, pickle
import os.path as osp 


# %%

# path = "//groups/hep/pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/data/dev_lvl7_mu_nu_e_classification_v003_unscaled.db"
# transform = "/groups/hep/johannbs/data/transformers2.pkl"
# with sqlite3.connect(path) as conn:
#     targets = pd.read_sql("select pid from truth", conn)


# %%
OscNext_path = osp.join("..", "data", "features", "OscNextCommonSplit_full")
OscNext_sample = pd.read_pickle(osp.join(OscNext_path, "train0.dat"))

MuonGun_path = osp.join("..", "data", "features", "MuonGun")
MuonGun_sample = pd.read_pickle(osp.join(MuonGun_path, "train0.dat"))
# MuonGun_sample2 = pd.read_pickle(osp.join(MuonGun_path, "train50000.dat"))


# %%
id = MuonGun_sample[1][:, 0]


# %%
with sqlite3.connect("/groups/hep/johannbs/data/rasmus_classification_muon_3neutrino_3mio.db") as conn:
    energy = pd.read_sql(f"select event_no, energy_log10 from truth where event_no in {tuple(id)}", conn).set_index("event_no", drop = True)

energy = energy.loc[id , :]

# %%
transformers = pd.read_pickle("/groups/hep/johannbs/data/transformers2.pkl")




# %%
pos_OscNext = np.concatenate(OscNext_sample[0])[:, :3]
pos_MuonGun = np.concatenate(MuonGun_sample[0])[:, :3]


# %%



# %%
fig, ax = plt.subplots(figsize = (5 , 8), nrows = 2, sharex = True)

# lims = ((-400, 400), ())

ax[0].set_title("x-z Distribution of Pulses in OscNext")
ax[0].hist2d(pos_OscNext[:, 0], pos_OscNext[:, 2], bins = 100)
ax[0].set_aspect("equal")
# ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("z [m]")



ax[1].set_title("x-z Distribution of Pulses in MuonGun")
ax[1].hist2d(pos_MuonGun[:, 0], pos_MuonGun[:, 2], bins = 100);
ax[1].set_ylabel("z [m]")
ax[1].set_xlabel("x [m]")

ax[1].set_aspect("equal")


# %%
fig.savefig(osp.join("..", "figures", "xz_distributions.pdf"))


# %%
Ns_OscNext = np.log10([x[:, -1].sum() for x in OscNext_sample[0]])


# %%
Es_OscNext = OscNext_sample[1][:, -1]


# %%
Ns_MuonGun = np.log10([x.shape[0] for x in MuonGun_sample[0]])
Es_MuonGun = np.array(energy)
Es_MuonGun = transformers['truth']["energy_log10"].inverse_transform(Es_MuonGun.reshape(1, -1)).flatten()

name_oscnext = ["OscNext" for i in range(len(Es_MuonGun))]
name_muongun = ["MuonGun" for i in range(len(Es_OscNext))]


dict = {
    "log(E)": np.concatenate([Es_OscNext, Es_MuonGun]),
    "log(N)": np.concatenate([Ns_OscNext, Ns_MuonGun]),
    "name":   name_oscnext + name_muongun
}

df = pd.DataFrame(dict)


# %%
plt.hist(Es_OscNext)





# %%
plt.hist(df['log(E)'][df.name == "MuonGun"], histtype = "step")
plt.hist(df['log(E)'][df.name == "OscNext"], histtype = "step")


# %%

from seaborn import jointplot
print("starting jointplot")
fig = jointplot(data = df, x = "log(N)", y = "log(E)", hue = "name", kind = "kde", xlim = (1., 3.), ylim = (0., 3.))



# %%
fig.savefig("../figures/pairplot_N_E.pdf")


# %%
targets.pid.value_counts()


# %%
len(targets)


# %%
targets.pid


# %%
transformers = pickle.load(open(transform, "rb"))
transformers["truth"].keys()


# %%
for col in targets.columns:
    if col in transformers["truth"].keys():
        targets[col] = transformers["truth"][col].inverse_transform(np.array(targets[col]).reshape(1, -1)).T


# %%
targets.hist("azimuth", bins = 100)


# %%
fig, ax = plt.subplots(figsize = (4, 9), nrows = 3)
ax = ax.flatten()

fig.suptitle("MuonGun Distributions")

ax = ax[0]
ax.hist(targets.energy_log10, bins = 100, histtype = "step")
ax.set_title(r"Energy $\log_{10}$ Distribution")
ax.set_xlabel(r"$\log_{10}(E)$")
ax.set_ylabel("Frequency")


ax = ax[1]
ax.hist(targets.azimuth, bins = 100, histtype = "step")
ax.set_title("Azimuth Distribution")
ax.set_xlabel("Azumuth [rad]")
ax.set_ylabel("Frequency")

ax = ax[2]
ax.hist(targets.zenith, bins = 100, histtype = "step")
ax.set_title("Zenith Distribution")
ax.set_xlabel("Zenith [rad]")
ax.set_ylabel("Frequency")


# %%



