import os

path="data/raw"

print(

    len(

      os.listdir(

       f"{path}/Parasitized"

      )

    )

)

print(

    len(

      os.listdir(

       f"{path}/Uninfected"

      )

    )

)