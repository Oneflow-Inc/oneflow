template <class T> class Maybe {};

Maybe<void> a() { return Maybe<void>(); }

int main() {
  a();
  return 0;
}
