# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # df = pd.read_csv("ga_convergence.csv")

# # # plt.plot(df["generation"], df["best_cost"]/1e7)
# # # plt.xlabel("Generation")
# # # plt.ylabel("Cost (₹ Cr)")
# # # plt.title("GA Convergence")
# # # plt.grid()

# # # plt.savefig("ga_convergence.png")
# # # plt.show()




# # # df = pd.read_csv("nsga2_pareto.csv")

# # # plt.scatter(df["total_cost"]/1e7, df["waste"])
# # # plt.xlabel("Cost (₹ Cr)")
# # # plt.ylabel("Waste")
# # # plt.title("Pareto: Cost vs Waste")

# # # plt.savefig("pareto_2d.png")
# # # plt.show()




# # from mpl_toolkits.mplot3d import Axes3D

# # # df = pd.read_csv("nsga2_pareto.csv")

# # # fig = plt.figure()
# # # ax = fig.add_subplot(111, projection='3d')

# # # ax.scatter(
# # #     df["total_cost"]/1e7,
# # #     df["waste"],
# # #     df["emissions"],
# # #     c=df["n_dcs_open"]
# # # )

# # # ax.set_xlabel("Cost (₹ Cr)")
# # # ax.set_ylabel("Waste")
# # # ax.set_zlabel("Emissions")

# # # plt.title("3D Pareto")

# # # plt.savefig("pareto_3d.png")
# # # plt.show()




# # # plt.scatter(df["n_dcs_open"], df["total_cost"]/1e7)

# # # plt.xlabel("DCs Opened")
# # # plt.ylabel("Cost (₹ Cr)")
# # # plt.title("Cost vs DCs")

# # # plt.savefig("cost_vs_dc.png")
# # # plt.show()


# # df = pd.read_csv("ga_best_dc_selection.csv")

# # selected = df[df["selected"] == 1]

# # plt.bar(selected["dc_id"], selected["selected"])
# # plt.xticks(rotation=90)

# # plt.title("Selected DCs")

# # plt.savefig("selected_dc.png")
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("ga_convergence.csv")

# plt.style.use("seaborn-v0_8")

# plt.figure(figsize=(8,5))
# plt.plot(df["generation"], df["best_cost"]/1e7, linewidth=2)

# plt.xlabel("Generation")
# plt.ylabel("Cost (₹ Cr)")
# plt.title("GA Convergence Curve")

# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn-v0_8")

# ======================
# LOAD DATA
# ======================
ga = pd.read_csv("ga_convergence.csv")
pareto = pd.read_csv("nsga2_pareto.csv")
dc = pd.read_csv("ga_best_dc_selection.csv")

# ======================
# CREATE DASHBOARD
# ======================
fig = plt.figure(figsize=(16,10))

# ----------------------
# 1. GA Convergence
# ----------------------
ax1 = plt.subplot(2,3,1)
ax1.plot(ga["generation"], ga["best_cost"]/1e7, linewidth=2)
ax1.set_title("GA Convergence")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Cost (₹ Cr)")
ax1.grid(alpha=0.3)

# ----------------------
# 2. Pareto (Cost vs Waste)
# ----------------------
ax2 = plt.subplot(2,3,2)
sc = ax2.scatter(
    pareto["total_cost"]/1e7,
    pareto["waste"],
    c=pareto["n_dcs_open"],
    cmap="viridis",
    s=60
)
ax2.set_title("Pareto: Cost vs Waste")
ax2.set_xlabel("Cost (₹ Cr)")
ax2.set_ylabel("Waste")
plt.colorbar(sc, ax=ax2, label="DCs")

# ----------------------
# 3. Cost vs DC
# ----------------------
ax3 = plt.subplot(2,3,3)
ax3.scatter(
    pareto["n_dcs_open"],
    pareto["total_cost"]/1e7,
    c=pareto["n_dcs_open"],
    cmap="coolwarm",
    s=60
)
ax3.set_title("Cost vs DCs")
ax3.set_xlabel("DCs Opened")
ax3.set_ylabel("Cost (₹ Cr)")
ax3.grid(alpha=0.3)

# ----------------------
# 4. 3D Pareto
# ----------------------
ax4 = fig.add_subplot(2,3,4, projection='3d')
p = ax4.scatter(
    pareto["total_cost"]/1e7,
    pareto["waste"],
    pareto["emissions"],
    c=pareto["n_dcs_open"],
    cmap="plasma"
)
ax4.set_xlabel("Cost")
ax4.set_ylabel("Waste")
ax4.set_zlabel("Emissions")
ax4.set_title("3D Pareto")

# ----------------------
# 5. DC Selection
# ----------------------
ax5 = plt.subplot(2,3,5)
selected = dc[dc["selected"] == 1]

ax5.barh(selected["dc_id"], selected["selected"])
ax5.set_title("Selected DCs")

# ----------------------
# 6. Cost Breakdown (dummy from GA final)
# ----------------------
ax6 = plt.subplot(2,3,6)

values = [31411128, 39170000, 4639347, 74541]
labels = ["Transport", "Fixed DC", "Disposal", "Carbon"]

ax6.pie(values, labels=labels, autopct="%1.1f%%")
ax6.set_title("Cost Breakdown")

plt.tight_layout()
plt.show()