#pragma once

// In a static link configuration we need to ensure the translation units containing
// the model registrations are included in the final binary. There are two parts to
// this solution:
// 
// 1. In the translation unit containing each model registration FORCE_LINK_ANCHOR
//    should be used to define a named symbol.
// 
// 2. In the target consuming the static library a corresponding FORCE_LINK_REQUIRE
//    should be used to pull that symbol (and the translation unit) in.

#if defined(_MSC_VER)
  #define FORCE_LINK_SYMBOL(name) __pragma(comment(linker, "/include:" #name))
#elif defined(__APPLE__)
  #define FORCE_LINK_SYMBOL(name)                                                                                      \
    extern "C" void name(void);                                                                                        \
    static void* _force_link_##name = (void*)&name;
#else // GCC/Linux clang
  #define FORCE_LINK_SYMBOL(name)                                                                                      \
    extern "C" void name(void);                                                                                        \
    static void* _force_link_##name __attribute__((used)) = (void*)&name;
#endif

// Declare a force-link anchor in a .cpp file inside the static lib
#define FORCE_LINK_ANCHOR(name)                                                                                        \
  extern "C" void _force_link_anchor_##name(void) {}

// Pull in that anchor from the consuming binary
#define FORCE_LINK_REQUIRE(name) FORCE_LINK_SYMBOL(_force_link_anchor_##name)
