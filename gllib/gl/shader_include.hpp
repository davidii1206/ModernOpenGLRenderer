#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace gl {

struct IncludeResult {
    std::string source;
    bool success = false;
    std::string error;
};

static IncludeResult read_file(std::string_view path) {
    std::ifstream file(path.data(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) return {{}, false, "cannot open"};
    auto size = file.tellg();
    std::string content(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(content.data(), size);
    return {std::move(content), true, {}};
}

// Evaluate a #if expression against a macro table.
// Supports: defined(X), !defined(X), numeric literals, macro substitution,
// &&, ||, ==, !=, parentheses, unary !
static bool eval_if_expr(const std::string& expr_, const std::map<std::string,int>& macros) {
    auto eval_primary = [&](const std::string& tok) -> int {
        std::string t = tok;
        t.erase(0, t.find_first_not_of(" \t"));
        t.erase(t.find_last_not_of(" \t") + 1);
        // Numeric literal?
        char* end = nullptr;
        long val = std::strtol(t.c_str(), &end, 0);
        if (end && *end == '\0') return (int)val;
        // Macro name
        auto it = macros.find(t);
        if (it != macros.end()) return it->second;
        return 0; // undefined macro → 0
    };

    // Pre-process: replace `defined(X)` with 1 or 0
    std::string s = expr_;
    {
        std::regex defined_re(R"(defined\s*\(\s*(\w+)\s*\))");
        std::smatch m;
        while (std::regex_search(s, m, defined_re)) {
            std::string sym = m[1].str();
            int val = macros.find(sym) != macros.end() ? 1 : 0;
            s.replace(m.position(0), m.length(0), std::to_string(val));
        }
    }

    // Tokenize: add spaces around operators and parens
    std::string tok_str;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (c == '(' || c == ')' || c == '!') {
            tok_str += ' ';
            tok_str += c;
            tok_str += ' ';
        } else {
            tok_str += c;
        }
    }

    std::vector<std::string> tokens;
    std::istringstream stream(tok_str);
    std::string token;
    while (stream >> token) tokens.push_back(token);

    if (tokens.empty()) return false;

    // Shunting-yard to evaluate the expression with known operators: !, &&, ||, ==, !=
    // Precedence: ! (highest), == !=, &&, || (lowest)
    enum Op { OP_NOT, OP_EQ, OP_NE, OP_AND, OP_OR, OP_NONE };
    auto op_prec = [](Op o) -> int {
        switch (o) {
            case OP_NOT: return 4;
            case OP_EQ:
            case OP_NE: return 3;
            case OP_AND: return 2;
            case OP_OR:  return 1;
            default: return 0;
        }
    };

    auto to_op = [](const std::string& t) -> Op {
        if (t == "!") return OP_NOT;
        if (t == "==") return OP_EQ;
        if (t == "!=") return OP_NE;
        if (t == "&&") return OP_AND;
        if (t == "||") return OP_OR;
        return OP_NONE;
    };

    std::vector<int> val_stack;
    std::vector<Op> op_stack;

    auto apply_op = [&](Op op) {
        if (op == OP_NOT) {
            if (val_stack.empty()) return;
            int v = val_stack.back(); val_stack.pop_back();
            val_stack.push_back(!v);
        } else {
            if (val_stack.size() < 2) return;
            int b = val_stack.back(); val_stack.pop_back();
            int a = val_stack.back(); val_stack.pop_back();
            switch (op) {
                case OP_EQ: val_stack.push_back(a == b ? 1 : 0); break;
                case OP_NE: val_stack.push_back(a != b ? 1 : 0); break;
                case OP_AND: val_stack.push_back(a && b ? 1 : 0); break;
                case OP_OR:  val_stack.push_back(a || b ? 1 : 0); break;
                default: val_stack.push_back(0); break;
            }
        }
    };

    for (size_t i = 0; i < tokens.size(); i++) {
        const std::string& tok = tokens[i];
        if (tok == "(") {
            op_stack.push_back(OP_NONE); // marker
        } else if (tok == ")") {
            while (!op_stack.empty() && op_stack.back() != OP_NONE) {
                apply_op(op_stack.back());
                op_stack.pop_back();
            }
            if (!op_stack.empty() && op_stack.back() == OP_NONE)
                op_stack.pop_back(); // pop the marker
        } else if (tok == "!") {
            // unary not
            op_stack.push_back(OP_NOT);
        } else if (tok == "&&" || tok == "||" || tok == "==" || tok == "!=") {
            Op op = to_op(tok);
            while (!op_stack.empty() && op_stack.back() != OP_NONE &&
                   op_prec(op_stack.back()) >= op_prec(op)) {
                apply_op(op_stack.back());
                op_stack.pop_back();
            }
            op_stack.push_back(op);
        } else {
            // primary
            int v = eval_primary(tok);
            val_stack.push_back(v);
        }
    }

    while (!op_stack.empty()) {
        apply_op(op_stack.back());
        op_stack.pop_back();
    }

    return !val_stack.empty() && val_stack.back() != 0;
}

// Strip inactive preprocessor branches from source given known macros.
// Handles #if / #ifdef / #ifndef / #elif / #else / #endif.
// Also tracks #define/#undef in the mutable macros map for active branches.
static std::string strip_inactive_branches(const std::string& source, std::map<std::string,int>& macros) {
    std::string result;
    std::istringstream stream(source);
    std::string line;

    std::vector<bool> active_stack = {true};
    std::vector<bool> taken_stack = {false};

    auto currently_active = [&]() -> bool {
        for (auto a : active_stack) if (!a) return false;
        return true;
    };

    while (std::getline(stream, line)) {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));

        // Track #define and #undef in active branches
        if (trimmed.rfind("#define ", 0) == 0 || trimmed.rfind("#define\t", 0) == 0) {
            if (currently_active()) {
                std::string def = trimmed.substr(8);
                // Skip function-like macros
                if (def.find('(') == std::string::npos) {
                    std::istringstream defs(def);
                    std::string name, val_str;
                    defs >> name;
                    if (!name.empty()) {
                        int val = 1;
                        if (defs >> val_str) {
                            char* end = nullptr;
                            long parsed = std::strtol(val_str.c_str(), &end, 0);
                            if (end && *end == '\0')
                                val = (int)parsed;
                        }
                        macros[name] = val;
                    }
                }
            }
            result += line + "\n";
            continue;
        }

        if (trimmed.rfind("#undef ", 0) == 0 || trimmed.rfind("#undef\t", 0) == 0) {
            if (currently_active()) {
                std::string name = trimmed.substr(7);
                name.erase(0, name.find_first_not_of(" \t"));
                if (!name.empty())
                    macros.erase(name);
            }
            result += line + "\n";
            continue;
        }

        if (trimmed.rfind("#if", 0) == 0) {
            bool condition = false;
            if (trimmed.rfind("#ifdef ", 0) == 0) {
                std::string sym = trimmed.substr(7);
                sym.erase(0, sym.find_first_not_of(" \t"));
                condition = macros.find(sym) != macros.end();
            } else if (trimmed.rfind("#ifndef ", 0) == 0) {
                std::string sym = trimmed.substr(8);
                sym.erase(0, sym.find_first_not_of(" \t"));
                condition = macros.find(sym) == macros.end();
            } else if (trimmed.rfind("#if ", 0) == 0) {
                std::string expr = trimmed.substr(4);
                condition = eval_if_expr(expr, macros);
            } else if (trimmed.size() >= 5 && trimmed[3] == '(') {
                // #if(expr)
                size_t paren = trimmed.rfind(')');
                if (paren != std::string::npos && paren > 4) {
                    std::string expr = trimmed.substr(4, paren - 4);
                    condition = eval_if_expr(expr, macros);
                } else {
                    condition = true;
                }
            } else {
                condition = true;
            }
            bool parent_active = currently_active();
            active_stack.push_back(parent_active && condition);
            taken_stack.push_back(parent_active && condition);
            continue;
        }

        if (trimmed.rfind("#elif", 0) == 0) {
            if (active_stack.size() >= 2) {
                bool parent_active = true;
                for (size_t i = 0; i < active_stack.size() - 1; i++) {
                    if (!active_stack[i]) { parent_active = false; break; }
                }
                bool taken_back = taken_stack.back();
                bool should_take = parent_active && !taken_back;
                bool elif_condition = false;
                if (should_take) {
                    std::string expr;
                    if (trimmed.size() > 6 && trimmed[5] == ' ') {
                        expr = trimmed.substr(6);
                    } else if (trimmed.size() > 5 && trimmed[5] == '(') {
                        size_t paren = trimmed.rfind(')');
                        if (paren != std::string::npos)
                            expr = trimmed.substr(6, paren - 6);
                    }
                    elif_condition = eval_if_expr(expr, macros);
                    active_stack.back() = elif_condition;
                    taken_stack.back() = elif_condition;
                } else {
                    active_stack.back() = false;
                }
            }
            // Never output #elif lines
            continue;
        }

        if (trimmed.rfind("#else", 0) == 0 &&
            (trimmed.size() == 5 || trimmed[5] == ' ' || trimmed[5] == '/')) {
            if (active_stack.size() >= 2) {
                bool parent_active = true;
                for (size_t i = 0; i < active_stack.size() - 1; i++) {
                    if (!active_stack[i]) { parent_active = false; break; }
                }
                bool taken_back = taken_stack.back();
                if (parent_active && !taken_back) {
                    active_stack.back() = true;
                } else {
                    active_stack.back() = false;
                }
            }
            continue;
        }

        if (trimmed.rfind("#endif", 0) == 0 &&
            (trimmed.size() == 6 || trimmed[6] == ' ' || trimmed[6] == '/' || trimmed[6] == ':')) {
            if (active_stack.size() > 1) {
                active_stack.pop_back();
                taken_stack.pop_back();
            }
            continue;
        }

        if (currently_active()) {
            result += line + "\n";
        }
    }

    return result;
}

static IncludeResult resolve_includes_impl(
    std::string_view source_path,
    const std::vector<std::string>& include_dirs,
    std::set<std::string>& included,
    std::map<std::string,int> macros)
{
    auto res = read_file(source_path);
    if (!res.success)
        return res;

    // Preprocess: strip inactive branches using known macros
    // strip_inactive_branches also tracks #define/#undef in macros
    std::string processed = strip_inactive_branches(res.source, macros);

    std::string result;
    static const std::regex re(R"(#\s*include\s+\"([^\"]+)\")");
    std::string remaining = processed;
    std::smatch m;
    std::string::const_iterator it = remaining.cbegin();
    std::string::const_iterator end = remaining.cend();

    while (std::regex_search(it, end, m, re)) {
        result.append(it, m[0].first);
        std::string inc_name = m[1].str();

        if (!included.insert(inc_name).second) {
            it = m[0].second;
            continue;
        }

        std::filesystem::path parent = std::filesystem::path(source_path).parent_path();
        bool found = false;
        auto try_resolve = [&](const std::filesystem::path& dir) -> bool {
            auto p = dir / inc_name;
            auto r = read_file(p.string());
            if (!r.success) return false;
            auto nested = resolve_includes_impl(p.string(), include_dirs, included, macros);
            if (!nested.success) {
                res = nested;
                return false;
            }
            result += nested.source;
            result += "\n";
            found = true;
            return true;
        };

        if (try_resolve(parent)) { /* ok */ }
        else if (!found) {
            for (auto& d : include_dirs) {
                if (try_resolve(d)) break;
            }
        }

        if (!found) {
            return {{}, false, "include not found: " + inc_name};
        }
        if (!res.success) return res;
        it = m[0].second;
    }
    result.append(it, end);
    return {std::move(result), true, {}};
}

inline IncludeResult resolve_includes(
    std::string_view source_path,
    const std::vector<std::string>& include_dirs = {},
    const std::map<std::string,int>& macros = {})
{
    std::set<std::string> included;
    std::map<std::string,int> mut = macros;
    return resolve_includes_impl(source_path, include_dirs, included, std::move(mut));
}

} // namespace gl
