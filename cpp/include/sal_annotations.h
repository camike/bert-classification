#pragma once

// 为MinGW定义微软SAL注解
#if defined(__MINGW32__) || defined(__MINGW64__)
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#ifndef _In_
#define _In_
#endif
#ifndef _In_opt_
#define _In_opt_
#endif
#ifndef _In_reads_bytes_
#define _In_reads_bytes_(x)
#endif
#ifndef _Inout_
#define _Inout_
#endif
#ifndef _Inout_opt_
#define _Inout_opt_
#endif
#ifndef _Inout_updates_bytes_
#define _Inout_updates_bytes_(x)
#endif
#ifndef _Out_
#define _Out_
#endif
#ifndef _Out_opt_
#define _Out_opt_
#endif
#ifndef _Outptr_
#define _Outptr_
#endif
#ifndef _Outptr_opt_
#define _Outptr_opt_
#endif
#ifndef _Outptr_opt_result_maybenull_
#define _Outptr_opt_result_maybenull_
#endif
#ifndef _Outptr_result_buffer_
#define _Outptr_result_buffer_(x)
#endif
#ifndef _Outptr_result_buffer_maybenull_
#define _Outptr_result_buffer_maybenull_(x)
#endif
#ifndef _Outptr_result_bytebuffer_
#define _Outptr_result_bytebuffer_(x)
#endif
#ifndef _Outptr_result_maybenull_
#define _Outptr_result_maybenull_
#endif
#ifndef _Outptr_result_maybenull_z_
#define _Outptr_result_maybenull_z_
#endif
#ifndef _Ret_maybenull_
#define _Ret_maybenull_
#endif
#ifndef _Ret_notnull_
#define _Ret_notnull_
#endif
#ifndef _Success_
#define _Success_(x)
#endif
#endif // defined(__MINGW32__) || defined(__MINGW64__) 