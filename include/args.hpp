#pragma once
#include <string>
#include <unordered_map>
#include <vector>

struct Args {
  std::unordered_map<std::string, std::string> kv;
  bool has(const std::string& key) const { return kv.find(key) != kv.end(); }
  std::string get(const std::string& key, const std::string& def="") const {
    auto it = kv.find(key);
    return (it == kv.end()) ? def : it->second;
  }
  int get_int(const std::string& key, int def=0) const {
    try { return std::stoi(get(key)); } catch (...) { return def; }
  }
  bool get_bool(const std::string& key, bool def=false) const {
    auto v = get(key);
    if (v.empty()) return def;
    return (v=="1" || v=="true" || v=="TRUE" || v=="on" || v=="ON");
  }
};

inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      s = s.substr(2);
      auto eq = s.find('=');
      if (eq != std::string::npos) {
        a.kv[s.substr(0, eq)] = s.substr(eq+1);
      } else {
        // flag or key value
        if (i+1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) {
          a.kv[s] = argv[++i];
        } else {
          a.kv[s] = "true";
        }
      }
    }
  }
  return a;
}
