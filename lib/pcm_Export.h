
#ifndef pcm_EXPORT_H
#define pcm_EXPORT_H

#ifdef pcm_BUILT_AS_STATIC
#  define pcm_EXPORT
#  define PCM_NO_EXPORT
#else
#  ifndef pcm_EXPORT
#    ifdef pcm_EXPORTS
        /* We are building this library */
#      define pcm_EXPORT 
#    else
        /* We are using this library */
#      define pcm_EXPORT 
#    endif
#  endif

#  ifndef PCM_NO_EXPORT
#    define PCM_NO_EXPORT 
#  endif
#endif

#ifndef PCM_DEPRECATED
#  define PCM_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef PCM_DEPRECATED_EXPORT
#  define PCM_DEPRECATED_EXPORT pcm_EXPORT PCM_DEPRECATED
#endif

#ifndef PCM_DEPRECATED_NO_EXPORT
#  define PCM_DEPRECATED_NO_EXPORT PCM_NO_EXPORT PCM_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef PCM_NO_DEPRECATED
#    define PCM_NO_DEPRECATED
#  endif
#endif

#endif
