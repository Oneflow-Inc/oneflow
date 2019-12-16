/*
* Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
* of a form of NVIDIA software license agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

#include "nvToolsExt.h"

#ifndef NVTOOLSEXT_SYNC_V3
#define NVTOOLSEXT_SYNC_V3

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* \cond SHOW_HIDDEN 
* \version \NVTX_VERSION_2
*/
#define NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(nvtxSyncUserAttributes_v0) ) )
/** \endcond */


/** 
* \page PAGE_SYNCHRONIZATION Synchronization
*
* This section covers a subset of the API that allow users to track additional
* synchronization details of their application.   Naming OS synchronization primitives 
* may allow users to better understand the data collected by traced synchronization 
* APIs.  Additionally, a user defined synchronization object can allow the users to
* to tell the tools when the user is building their own synchronization system
* that do not rely on the OS to provide behaviors and instead use techniques like
* atomic operations and spinlocks.  
*
* See module \ref SYNCHRONIZATION for details.
*
* \par Example:
* \code
* class MyMutex
* {
*     volatile long bLocked;
*     nvtxSyncUser_t hSync;
* public:
*     MyMutex(const char* name, nvtxDomainHandle_t d){
*          bLocked = 0;
*
*          nvtxSyncUserAttributes_t attribs = { 0 };
*          attribs.version = NVTX_VERSION;
*          attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
*          attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
*          attribs.message.ascii = name;
*          hSync = nvtxDomainSyncUserCreate(d, &attribs);
*     }
*
*     ~MyMutex() {
*          nvtxDomainSyncUserDestroy(hSync);
*     }
*
*     bool Lock() {
*          nvtxDomainSyncUserAcquireStart(hSync);
*          bool acquired = __sync_bool_compare_and_swap(&bLocked, 0, 1);//atomic compiler intrinsic 

*          if (acquired) {
*              nvtxDomainSyncUserAcquireSuccess(hSync);
*          }
*          else {
*              nvtxDomainSyncUserAcquireFailed(hSync);
*          }
*          return acquired;
*     }

*     void Unlock() {
*          nvtxDomainSyncUserReleasing(hSync);
*          bLocked = false;
*     }
* };
* \endcode
* 
* \version \NVTX_VERSION_2
*/

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN 
* \brief Used to build a non-colliding value for resource types separated class
* \version \NVTX_VERSION_2
*/
#define NVTX_RESOURCE_CLASS_SYNC_OS 2 /**< Synchronization objects that are OS specific. */
#define NVTX_RESOURCE_CLASS_SYNC_PTHREAD 3 /**< Synchronization objects that are from the POSIX Threads API (pthread)*/
/** \endcond */


/*  ------------------------------------------------------------------------- */
/** \defgroup SYNCHRONIZATION Synchronization
* See page \ref PAGE_SYNCHRONIZATION.
* @{
*/

/** \brief Resource type values for OSs with POSIX Thread API support
 */
typedef enum nvtxResourceSyncPosixThreadType_t
{
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 1), /* pthread_mutex_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 2), /* pthread_cond_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 3), /* pthread_rwlock_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 4), /* pthread_barrier_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 5), /* pthread_spinlock_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 6) /* pthread_once_t  */
} nvtxResourceSyncPosixThreadType_t;

/** \brief Resource type values for Windows OSs
*/
typedef enum nvtxResourceSyncWindowsType_t
{
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_SEMAPHORE = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_EVENT = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_CRITICAL_SECTION = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_SRWLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5)
} nvtxResourceSyncWindowsType_t;

/** \brief Resource type values for Linux and Linux derived OSs such as Android
* \sa
* ::nvtxResourceSyncPosixThreadType_t
*/
typedef enum nvtxResourceSyncLinuxType_t
{
    NVTX_RESOURCE_TYPE_SYNC_LINUX_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_FUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SEMAPHORE = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_COMPLETION = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SPINLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SEQLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 6),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_RCU = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 7)
} nvtxResourceSyncLinuxType_t;

/** \brief Resource type values for Android come from Linux.
* \sa
* ::nvtxResourceSyncLinuxType_t
* ::nvtxResourceSyncPosixThreadType_t
*/
typedef enum nvtxResourceSyncLinuxType_t nvtxResourceSyncAndroidType_t;

/** \brief User Defined Synchronization Object Handle .
* \anchor SYNCUSER_HANDLE_STRUCTURE
*
* This structure is opaque to the user and is used as a handle to reference
* a user defined syncrhonization object.  The tools will return a pointer through the API for the application
* to hold on it's behalf to reference the string in the future.
*
*/
typedef struct nvtxSyncUser* nvtxSyncUser_t;

/** \brief User Defined Synchronization Object Attributes Structure.
* \anchor USERDEF_SYNC_ATTRIBUTES_STRUCTURE
*
* This structure is used to describe the attributes of a user defined synchronization 
* object.  The layout of the structure is defined by a specific version of the tools 
* extension library and can change between different versions of the Tools Extension
* library.
*
* \par Initializing the Attributes
*
* The caller should always perform the following three tasks when using
* attributes:
* <ul>
*    <li>Zero the structure
*    <li>Set the version field
*    <li>Set the size field
* </ul>
*
* Zeroing the structure sets all the event attributes types and values
* to the default value.
*
* The version and size field are used by the Tools Extension
* implementation to handle multiple versions of the attributes structure.
*
* It is recommended that the caller use one of the following to methods
* to initialize the event attributes structure:
*
* \par Method 1: Initializing nvtxEventAttributes for future compatibility
* \code
* nvtxSyncUserAttributes_t attribs = {0};
* attribs.version = NVTX_VERSION;
* attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
* \endcode
*
* \par Method 2: Initializing nvtxSyncUserAttributes_t for a specific version
* \code
* nvtxSyncUserAttributes_t attribs = {0};
* attribs.version = 1;
* attribs.size = (uint16_t)(sizeof(nvtxSyncUserAttributes_t));
* \endcode
*
* If the caller uses Method 1 it is critical that the entire binary
* layout of the structure be configured to 0 so that all fields
* are initialized to the default value.
*
* The caller should either use both NVTX_VERSION and
* NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
* and a versioned type (Method 2).  Using a mix of the two methods
* will likely cause either source level incompatibility or binary
* incompatibility in the future.
*
* \par Settings Attribute Types and Values
*
*
* \par Example:
* \code
* // Initialize
* nvtxSyncUserAttributes_t attribs = {0};
* attribs.version = NVTX_VERSION;
* attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
*
* // Configure the Attributes
* attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
* attribs.message.ascii = "Example";
* \endcode
*
* \sa
* ::nvtxDomainSyncUserCreate
*/
typedef struct nvtxSyncUserAttributes_v0
{
    /**
    * \brief Version flag of the structure.
    *
    * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
    * supported in this header file. This can optionally be overridden to
    * another version of the tools extension library.
    */
    uint16_t version;

    /**
    * \brief Size of the structure.
    *
    * Needs to be set to the size in bytes of the event attribute
    * structure used to specify the event.
    */
    uint16_t size;

    /** \brief Message type specified in this attribute structure.
    *
    * Defines the message format of the attribute structure's \ref nvtxSyncUserAttributes_v0::message
    * "message" field.
    *
    * Default Value is NVTX_MESSAGE_UNKNOWN
    */
    int32_t messageType;            /* nvtxMessageType_t */

    /** \brief Message assigned to this attribute structure.
    *
    * The text message that is attached to an event.
    */
    nvtxMessageValue_t message;

} nvtxSyncUserAttributes_v0;

typedef struct nvtxSyncUserAttributes_v0 nvtxSyncUserAttributes_t;

/* ------------------------------------------------------------------------- */
/** \brief Create a user defined synchronization object 
* This is used to track non-OS synchronization working with spinlocks and atomics
*
* \param domain - Domain to own the resource
* \param attribs - A structure to assign multiple attributes to the object.
*
* \return A handle that represents the newly created user defined synchronization object.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/
NVTX_DECLSPEC nvtxSyncUser_t NVTX_API nvtxDomainSyncUserCreate(nvtxDomainHandle_t domain, const nvtxSyncUserAttributes_t* attribs);

/* ------------------------------------------------------------------------- */
/** \brief Destroy a user defined synchronization object
* This is used to track non-OS synchronization working with spinlocks and atomics
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/
NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserDestroy(nvtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools that an attempt to acquire a user defined synchronization object
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/
NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireStart(nvtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of failure in acquiring a user defined synchronization object
* This should be called after \ref nvtxDomainSyncUserAcquireStart
* 
* \param handle - A handle to the object to operate on.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireFailed(nvtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of success in acquiring a user defined synchronization object
* This should be called after \ref nvtxDomainSyncUserAcquireStart.
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireSuccess(nvtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of releasing a reservation on user defined synchronization object
* This should be called after \ref nvtxDomainSyncUserAcquireSuccess.
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::nvtxDomainSyncUserCreate
* ::nvtxDomainSyncUserDestroy
* ::nvtxDomainSyncUserAcquireStart
* ::nvtxDomainSyncUserAcquireFailed
* ::nvtxDomainSyncUserAcquireSuccess
* ::nvtxDomainSyncUserReleasing
*
* \version \NVTX_VERSION_2
*/
NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserReleasing(nvtxSyncUser_t handle);


/** @} */ /*END defgroup*/

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef NVTX_NO_IMPL
#define NVTX_IMPL_GUARD_SYNC /* Ensure other headers cannot included directly */
#include "nvtxDetail/nvtxImplSync_v3.h"
#undef NVTX_IMPL_GUARD_SYNC
#endif /*NVTX_NO_IMPL*/

#endif /* NVTOOLSEXT_SYNC_V3 */
