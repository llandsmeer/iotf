set -ex
pdoc3 ioperf --html --force --output-dir /tmp
rm -fr docs
mv /tmp/ioperf/ docs
tree docs
