
if [ "${BUILD_WITH_SERIALIZATION_WO_PYTHON}" = "true" ]; then
  cmake -DWITH_SERIALIZATION=ON -DWITH_PYTHON=OFF -DEIGEN3_INCLUDE_DIR=`pwd`/eigen-eigen-bdd17ee3b1b3/ ..
else
  cmake -DWITH_SERIALIZATION=OFF -DWITH_PYTHON=ON -DEIGEN3_INCLUDE_DIR=`pwd`/eigen-eigen-bdd17ee3b1b3/ ..
fi
