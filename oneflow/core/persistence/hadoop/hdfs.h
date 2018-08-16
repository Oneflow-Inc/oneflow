/********************************************************************
 * 2014 -
 * open source under Apache License Version 2.0
 ********************************************************************/
/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _HDFS_LIBHDFS3_CLIENT_HDFS_H_
#define _HDFS_LIBHDFS3_CLIENT_HDFS_H_

#include <errno.h>  /* for EINTERNAL, etc. */
#include <fcntl.h>  /* for O_RDONLY, O_WRONLY */
#include <stdint.h> /* for uint64_t, etc. */
#include <time.h>   /* for time_t */

#ifndef O_RDONLY
#define O_RDONLY 1
#endif

#ifndef O_WRONLY
#define O_WRONLY 2
#endif

#ifndef EINTERNAL
#define EINTERNAL 255
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: DEPRECATED is not supported by the compiler.")
#define DEPRECATED
#endif

/** All APIs set errno to meaningful values */

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Some utility decls used in libhdfs.
 */
typedef int32_t tSize;    /// size of data for read/write io ops
typedef time_t tTime;     /// time type in seconds
typedef int64_t tOffset;  /// offset within the file
typedef uint16_t tPort;   /// port

typedef enum tObjectKind {
  kObjectKindFile = 'F',
  kObjectKindDirectory = 'D',
} tObjectKind;

struct HdfsFileSystemInternalWrapper;
typedef struct HdfsFileSystemInternalWrapper *hdfsFS;

struct HdfsFileInternalWrapper;
typedef struct HdfsFileInternalWrapper *hdfsFile;

struct hdfsBuilder;

/**
 * Return error information of last failed operation.
 *
 * @return 			A not NULL const string point of last error information.
 * 					Caller can only read this message and keep it unchanged. No
 * need to free it. If last operation finished successfully, the returned message is undefined.
 */
const char *hdfsGetLastError();

/**
 * Determine if a file is open for read.
 *
 * @param file     The HDFS file
 * @return         1 if the file is open for read; 0 otherwise
 */
int hdfsFileIsOpenForRead(hdfsFile file);

/**
 * Determine if a file is open for write.
 *
 * @param file     The HDFS file
 * @return         1 if the file is open for write; 0 otherwise
 */
int hdfsFileIsOpenForWrite(hdfsFile file);

/**
 * hdfsConnectAsUser - Connect to a hdfs file system as a specific user
 * Connect to the hdfs.
 * @param nn   The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port The port on which the server is listening.
 * @param user the user name (this is hadoop domain user). Or NULL is equivelant to
 * hhdfsConnect(host, port)
 * @return Returns a handle to the filesystem or NULL on error.
 * @deprecated Use hdfsBuilderConnect instead.
 */
hdfsFS hdfsConnectAsUser(const char *nn, tPort port, const char *user);

/**
 * hdfsConnect - Connect to a hdfs file system.
 * Connect to the hdfs.
 * @param nn   The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port The port on which the server is listening.
 * @return Returns a handle to the filesystem or NULL on error.
 * @deprecated Use hdfsBuilderConnect instead.
 */
hdfsFS hdfsConnect(const char *nn, tPort port);

/**
 * hdfsConnect - Connect to an hdfs file system.
 *
 * Forces a new instance to be created
 *
 * @param nn     The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port   The port on which the server is listening.
 * @param user   The user name to use when connecting
 * @return       Returns a handle to the filesystem or NULL on error.
 * @deprecated   Use hdfsBuilderConnect instead.
 */
hdfsFS hdfsConnectAsUserNewInstance(const char *nn, tPort port, const char *user);

/**
 * hdfsConnect - Connect to an hdfs file system.
 *
 * Forces a new instance to be created
 *
 * @param nn     The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port   The port on which the server is listening.
 * @return       Returns a handle to the filesystem or NULL on error.
 * @deprecated   Use hdfsBuilderConnect instead.
 */
hdfsFS hdfsConnectNewInstance(const char *nn, tPort port);

/**
 * Connect to HDFS using the parameters defined by the builder.
 *
 * The HDFS builder will be freed, whether or not the connection was
 * successful.
 *
 * Every successful call to hdfsBuilderConnect should be matched with a call
 * to hdfsDisconnect, when the hdfsFS is no longer needed.
 *
 * @param bld    The HDFS builder
 * @return       Returns a handle to the filesystem, or NULL on error.
 */
hdfsFS hdfsBuilderConnect(struct hdfsBuilder *bld);

/**
 * Create an HDFS builder.
 *
 * @return The HDFS builder, or NULL on error.
 */
struct hdfsBuilder *hdfsNewBuilder(void);

/**
 * Do nothing, we always create a new instance
 *
 * @param bld The HDFS builder
 */
void hdfsBuilderSetForceNewInstance(struct hdfsBuilder *bld);

/**
 * Set the HDFS NameNode to connect to.
 *
 * @param bld  The HDFS builder
 * @param nn   The NameNode to use.
 *
 *             If the string given is 'default', the default NameNode
 *             configuration will be used (from the XML configuration files)
 *
 *             If NULL is given, a LocalFileSystem will be created.
 *
 *             If the string starts with a protocol type such as file:// or
 *             hdfs://, this protocol type will be used.  If not, the
 *             hdfs:// protocol type will be used.
 *
 *             You may specify a NameNode port in the usual way by
 *             passing a string of the format hdfs://<hostname>:<port>.
 *             Alternately, you may set the port with
 *             hdfsBuilderSetNameNodePort.  However, you must not pass the
 *             port in two different ways.
 */
void hdfsBuilderSetNameNode(struct hdfsBuilder *bld, const char *nn);

/**
 * Set the port of the HDFS NameNode to connect to.
 *
 * @param bld The HDFS builder
 * @param port The port.
 */
void hdfsBuilderSetNameNodePort(struct hdfsBuilder *bld, tPort port);

/**
 * Set the username to use when connecting to the HDFS cluster.
 *
 * @param bld The HDFS builder
 * @param userName The user name.  The string will be shallow-copied.
 */
void hdfsBuilderSetUserName(struct hdfsBuilder *bld, const char *userName);

/**
 * Set the path to the Kerberos ticket cache to use when connecting to
 * the HDFS cluster.
 *
 * @param bld The HDFS builder
 * @param kerbTicketCachePath The Kerberos ticket cache path.  The string
 *                            will be shallow-copied.
 */
void hdfsBuilderSetKerbTicketCachePath(struct hdfsBuilder *bld, const char *kerbTicketCachePath);

/**
 * Set the token used to authenticate
 *
 * @param bld The HDFS builder
 * @param token The token used to authenticate
 */
void hdfsBuilderSetToken(struct hdfsBuilder *bld, const char *token);

/**
 * Free an HDFS builder.
 *
 * It is normally not necessary to call this function since
 * hdfsBuilderConnect frees the builder.
 *
 * @param bld The HDFS builder
 */
void hdfsFreeBuilder(struct hdfsBuilder *bld);

/**
 * Set a configuration string for an HdfsBuilder.
 *
 * @param key      The key to set.
 * @param val      The value, or NULL to set no value.
 *                 This will be shallow-copied.  You are responsible for
 *                 ensuring that it remains valid until the builder is
 *                 freed.
 *
 * @return         0 on success; nonzero error code otherwise.
 */
int hdfsBuilderConfSetStr(struct hdfsBuilder *bld, const char *key, const char *val);

/**
 * Get a configuration string.
 *
 * @param key      The key to find
 * @param val      (out param) The value.  This will be set to NULL if the
 *                 key isn't found.  You must free this string with
 *                 hdfsConfStrFree.
 *
 * @return         0 on success; nonzero error code otherwise.
 *                 Failure to find the key is not an error.
 */
int hdfsConfGetStr(const char *key, char **val);

/**
 * Get a configuration integer.
 *
 * @param key      The key to find
 * @param val      (out param) The value.  This will NOT be changed if the
 *                 key isn't found.
 *
 * @return         0 on success; nonzero error code otherwise.
 *                 Failure to find the key is not an error.
 */
int hdfsConfGetInt(const char *key, int32_t *val);

/**
 * Free a configuration string found with hdfsConfGetStr.
 *
 * @param val      A configuration string obtained from hdfsConfGetStr
 */
void hdfsConfStrFree(char *val);

/**
 * hdfsDisconnect - Disconnect from the hdfs file system.
 * Disconnect from hdfs.
 * @param fs The configured filesystem handle.
 * @return Returns 0 on success, -1 on error.
 *         Even if there is an error, the resources associated with the
 *         hdfsFS will be freed.
 */
int hdfsDisconnect(hdfsFS fs);

/**
 * hdfsOpenFile - Open a hdfs file in given mode.
 * @param fs The configured filesystem handle.
 * @param path The full path to the file.
 * @param flags - an | of bits/fcntl.h file flags - supported flags are O_RDONLY, O_WRONLY (meaning
 * create or overwrite i.e., implies O_TRUNCAT), O_WRONLY|O_APPEND and O_SYNC. Other flags are
 * generally ignored other than (O_RDWR || (O_EXCL & O_CREAT)) which return NULL and set errno equal
 * ENOTSUP.
 * @param bufferSize Size of buffer for read/write - pass 0 if you want
 * to use the default configured values.
 * @param replication Block replication - pass 0 if you want to use
 * the default configured values.
 * @param blocksize Size of block - pass 0 if you want to use the
 * default configured values.
 * @return Returns the handle to the open file or NULL on error.
 */
hdfsFile hdfsOpenFile(hdfsFS fs, const char *path, int flags, int bufferSize, short replication,
                      tOffset blocksize);

/**
 * hdfsCloseFile - Close an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns 0 on success, -1 on error.
 *         On error, errno will be set appropriately.
 *         If the hdfs file was valid, the memory associated with it will
 *         be freed at the end of this call, even if there was an I/O
 *         error.
 */
int hdfsCloseFile(hdfsFS fs, hdfsFile file);

/**
 * hdfsExists - Checks if a given path exsits on the filesystem
 * @param fs The configured filesystem handle.
 * @param path The path to look for
 * @return Returns 0 on success, -1 on error.
 */
int hdfsExists(hdfsFS fs, const char *path);

/**
 * hdfsSeek - Seek to given offset in file.
 * This works only for files opened in read-only mode.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param desiredPos Offset into the file to seek into.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsSeek(hdfsFS fs, hdfsFile file, tOffset desiredPos);

/**
 * hdfsTell - Get the current offset in the file, in bytes.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Current offset, -1 on error.
 */
tOffset hdfsTell(hdfsFS fs, hdfsFile file);

/**
 * hdfsRead - Read data from an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param buffer The buffer to copy read bytes into.
 * @param length The length of the buffer.
 * @return      On success, a positive number indicating how many bytes
 *              were read.
 *              On end-of-file, 0.
 *              On error, -1.  Errno will be set to the error code.
 *              Just like the POSIX read function, hdfsRead will return -1
 *              and set errno to EINTR if data is temporarily unavailable,
 *              but we are not yet at the end of the file.
 */
tSize hdfsRead(hdfsFS fs, hdfsFile file, void *buffer, tSize length);

/**
 * hdfsPread - Positional read of data from an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param offset Position from which to read
 * @param buffer The buffer to copy read bytes into.
 * @param length The length of the buffer.
 * @return      See hdfsRead
 */
tSize hdfsPread(hdfsFS fs, hdfsFile file, tOffset offset, void *buffer, tSize length);

/**
 * hdfsWrite - Write data into an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param buffer The data.
 * @param length The no. of bytes to write.
 * @return Returns the number of bytes written, -1 on error.
 */
tSize hdfsWrite(hdfsFS fs, hdfsFile file, const void *buffer, tSize length);

/**
 * hdfsWrite - Flush the data.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsFlush(hdfsFS fs, hdfsFile file);

/**
 * hdfsHFlush - Flush out the data in client's user buffer. After the
 * return of this call, new readers will see the data.
 * @param fs configured filesystem handle
 * @param file file handle
 * @return 0 on success, -1 on error and sets errno
 */
int hdfsHFlush(hdfsFS fs, hdfsFile file);

/**
 * This function is deprecated. Please use hdfsHSync instead.
 *
 * hdfsSync - Flush out and sync the data in client's user buffer. After the
 * return of this call, new readers will see the data.
 * @param fs configured filesystem handle
 * @param file file handle
 * @return 0 on success, -1 on error and sets errno
 */
DEPRECATED int hdfsSync(hdfsFS fs, hdfsFile file);

/**
 * hdfsHSync - Flush out and sync the data in client's user buffer. After the
 * return of this call, new readers will see the data.
 * @param fs configured filesystem handle
 * @param file file handle
 * @return 0 on success, -1 on error and sets errno
 */
int hdfsHSync(hdfsFS fs, hdfsFile file);

/**
 * hdfsAvailable - Number of bytes that can be read from this
 * input stream without blocking.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns available bytes; -1 on error.
 */
int hdfsAvailable(hdfsFS fs, hdfsFile file);

/**
 * hdfsCopy - Copy file from one filesystem to another.
 * @param srcFS The handle to source filesystem.
 * @param src The path of source file.
 * @param dstFS The handle to destination filesystem.
 * @param dst The path of destination file.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsCopy(hdfsFS srcFS, const char *src, hdfsFS dstFS, const char *dst);

/**
 * hdfsMove - Move file from one filesystem to another.
 * @param srcFS The handle to source filesystem.
 * @param src The path of source file.
 * @param dstFS The handle to destination filesystem.
 * @param dst The path of destination file.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsMove(hdfsFS srcFS, const char *src, hdfsFS dstFS, const char *dst);

/**
 * hdfsDelete - Delete file.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @param recursive if path is a directory and set to
 * non-zero, the directory is deleted else throws an exception. In
 * case of a file the recursive argument is irrelevant.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsDelete(hdfsFS fs, const char *path, int recursive);

/**
 * hdfsRename - Rename file.
 * @param fs The configured filesystem handle.
 * @param oldPath The path of the source file.
 * @param newPath The path of the destination file.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsRename(hdfsFS fs, const char *oldPath, const char *newPath);

/**
 * hdfsGetWorkingDirectory - Get the current working directory for
 * the given filesystem.
 * @param fs The configured filesystem handle.
 * @param buffer The user-buffer to copy path of cwd into.
 * @param bufferSize The length of user-buffer.
 * @return Returns buffer, NULL on error.
 */
char *hdfsGetWorkingDirectory(hdfsFS fs, char *buffer, size_t bufferSize);

/**
 * hdfsSetWorkingDirectory - Set the working directory. All relative
 * paths will be resolved relative to it.
 * @param fs The configured filesystem handle.
 * @param path The path of the new 'cwd'.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsSetWorkingDirectory(hdfsFS fs, const char *path);

/**
 * hdfsCreateDirectory - Make the given file and all non-existent
 * parents into directories.
 * @param fs The configured filesystem handle.
 * @param path The path of the directory.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsCreateDirectory(hdfsFS fs, const char *path);

/**
 * hdfsSetReplication - Set the replication of the specified
 * file to the supplied value
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @return Returns 0 on success, -1 on error.
 */
int hdfsSetReplication(hdfsFS fs, const char *path, int16_t replication);

/**
 * hdfsEncryptionZoneInfo- Information about an encryption zone.
 */
typedef struct {
  int mSuite;                 /* the suite of encryption zone */
  int mCryptoProtocolVersion; /* the version of crypto protocol */
  int64_t mId;                /* the id of encryption zone */
  char *mPath;                /* the path of encryption zone */
  char *mKeyName;             /* the key name of encryption zone */
} hdfsEncryptionZoneInfo;

/**
 * hdfsEncryptionFileInfo - Information about an encryption file/directory.
 */
typedef struct {
  int mSuite;                 /* the suite of encryption file/directory */
  int mCryptoProtocolVersion; /* the version of crypto protocol */
  char *mKey;                 /* the key of encryption file/directory */
  char *mKeyName;             /* the key name of encryption file/directory */
  char *mIv;                  /* the iv of encryption file/directory */
  char *mEzKeyVersionName;    /* the version encryption file/directory */
} hdfsEncryptionFileInfo;

/**
 * hdfsFileInfo - Information about a file/directory.
 */
typedef struct {
  tObjectKind mKind;  /* file or directory */
  char *mName;        /* the name of the file */
  tTime mLastMod;     /* the last modification time for the file in seconds */
  tOffset mSize;      /* the size of the file in bytes */
  short mReplication; /* the count of replicas */
  tOffset mBlockSize; /* the block size for the file */
  char *mOwner;       /* the owner of the file */
  char *mGroup;       /* the group associated with the file */
  short mPermissions; /* the permissions associated with the file */
  tTime mLastAccess;  /* the last access time for the file in seconds */
  hdfsEncryptionFileInfo *mHdfsEncryptionFileInfo; /* the encryption info of the file/directory */
} hdfsFileInfo;

/**
 * hdfsListDirectory - Get list of files/directories for a given
 * directory-path. hdfsFreeFileInfo should be called to deallocate memory.
 * @param fs The configured filesystem handle.
 * @param path The path of the directory.
 * @param numEntries Set to the number of files/directories in path.
 * @return Returns a dynamically-allocated array of hdfsFileInfo
 * objects; NULL on error.
 */
hdfsFileInfo *hdfsListDirectory(hdfsFS fs, const char *path, int *numEntries);

/**
 * hdfsGetPathInfo - Get information about a path as a (dynamically
 * allocated) single hdfsFileInfo struct. hdfsFreeFileInfo should be
 * called when the pointer is no longer needed.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @return Returns a dynamically-allocated hdfsFileInfo object;
 * NULL on error.
 */
hdfsFileInfo *hdfsGetPathInfo(hdfsFS fs, const char *path);

/**
 * hdfsFreeFileInfo - Free up the hdfsFileInfo array (including fields)
 * @param infos The array of dynamically-allocated hdfsFileInfo
 * objects.
 * @param numEntries The size of the array.
 */
void hdfsFreeFileInfo(hdfsFileInfo *infos, int numEntries);

/**
 * hdfsFreeEncryptionZoneInfo - Free up the hdfsEncryptionZoneInfo array (including fields)
 * @param infos The array of dynamically-allocated hdfsEncryptionZoneInfo
 * objects.
 * @param numEntries The size of the array.
 */
void hdfsFreeEncryptionZoneInfo(hdfsEncryptionZoneInfo *infos, int numEntries);

/**
 * hdfsGetHosts - Get hostnames where a particular block (determined by
 * pos & blocksize) of a file is stored. The last element in the array
 * is NULL. Due to replication, a single block could be present on
 * multiple hosts.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @param start The start of the block.
 * @param length The length of the block.
 * @return Returns a dynamically-allocated 2-d array of blocks-hosts;
 * NULL on error.
 */
char ***hdfsGetHosts(hdfsFS fs, const char *path, tOffset start, tOffset length);

/**
 * hdfsFreeHosts - Free up the structure returned by hdfsGetHosts
 * @param hdfsFileInfo The array of dynamically-allocated hdfsFileInfo
 * objects.
 * @param numEntries The size of the array.
 */
void hdfsFreeHosts(char ***blockHosts);

/**
 * hdfsGetDefaultBlockSize - Get the default blocksize.
 *
 * @param fs            The configured filesystem handle.
 * @deprecated          Use hdfsGetDefaultBlockSizeAtPath instead.
 *
 * @return              Returns the default blocksize, or -1 on error.
 */
tOffset hdfsGetDefaultBlockSize(hdfsFS fs);

/**
 * hdfsGetCapacity - Return the raw capacity of the filesystem.
 * @param fs The configured filesystem handle.
 * @return Returns the raw-capacity; -1 on error.
 */
tOffset hdfsGetCapacity(hdfsFS fs);

/**
 * hdfsGetUsed - Return the total raw size of all files in the filesystem.
 * @param fs The configured filesystem handle.
 * @return Returns the total-size; -1 on error.
 */
tOffset hdfsGetUsed(hdfsFS fs);

/**
 * Change the user and/or group of a file or directory.
 *
 * @param fs            The configured filesystem handle.
 * @param path          the path to the file or directory
 * @param owner         User string.  Set to NULL for 'no change'
 * @param group         Group string.  Set to NULL for 'no change'
 * @return              0 on success else -1
 */
int hdfsChown(hdfsFS fs, const char *path, const char *owner, const char *group);

/**
 * hdfsChmod
 * @param fs The configured filesystem handle.
 * @param path the path to the file or directory
 * @param mode the bitmask to set it to
 * @return 0 on success else -1
 */
int hdfsChmod(hdfsFS fs, const char *path, short mode);

/**
 * hdfsUtime
 * @param fs The configured filesystem handle.
 * @param path the path to the file or directory
 * @param mtime new modification time or -1 for no change
 * @param atime new access time or -1 for no change
 * @return 0 on success else -1
 */
int hdfsUtime(hdfsFS fs, const char *path, tTime mtime, tTime atime);

/**
 * hdfsTruncate - Truncate the file in the indicated path to the indicated size.
 * @param fs The configured filesystem handle.
 * @param path the path to the file.
 * @param pos the position the file will be truncated to.
 * @param shouldWait output value, true if and client does not need to wait for block recovery,
 * false if client needs to wait for block recovery.
 */
int hdfsTruncate(hdfsFS fs, const char *path, tOffset pos, int *shouldWait);

/**
 * Get a delegation token from namenode.
 * The token should be freed using hdfsFreeDelegationToken after canceling the token or token
 * expired.
 *
 * @param fs The file system
 * @param renewer The user who will renew the token
 *
 * @return Return a delegation token, NULL on error.
 */
char *hdfsGetDelegationToken(hdfsFS fs, const char *renewer);

/**
 * Free a delegation token.
 *
 * @param token The token to be freed.
 */
void hdfsFreeDelegationToken(char *token);

/**
 * Renew a delegation token.
 *
 * @param fs The file system.
 * @param token The token to be renewed.
 *
 * @return the new expiration time
 */
int64_t hdfsRenewDelegationToken(hdfsFS fs, const char *token);

/**
 * Cancel a delegation token.
 *
 * @param fs The file system.
 * @param token The token to be canceled.
 *
 * @return return 0 on success, -1 on error.
 */
int hdfsCancelDelegationToken(hdfsFS fs, const char *token);

typedef struct Namenode {
  char *rpc_addr;   // namenode rpc address and port, such as "host:8020"
  char *http_addr;  // namenode http address and port, such as "host:50070"
} Namenode;

/**
 * If hdfs is configured with HA namenode, return all namenode informations as an array.
 * Else return NULL.
 *
 * Using configure file which is given by environment parameter LIBHDFS3_CONF
 * or "hdfs-client.xml" in working directory.
 *
 * @param nameservice hdfs name service id.
 * @param size output the size of returning array.
 *
 * @return return an array of all namenode information.
 */
Namenode *hdfsGetHANamenodes(const char *nameservice, int *size);

/**
 * If hdfs is configured with HA namenode, return all namenode informations as an array.
 * Else return NULL.
 *
 * @param conf the path of configure file.
 * @param nameservice hdfs name service id.
 * @param size output the size of returning array.
 *
 * @return return an array of all namenode information.
 */
Namenode *hdfsGetHANamenodesWithConfig(const char *conf, const char *nameservice, int *size);

/**
 * Free the array returned by hdfsGetConfiguredNamenodes()
 *
 * @param the array return by hdfsGetConfiguredNamenodes()
 */
void hdfsFreeNamenodeInformation(Namenode *namenodes, int size);

typedef struct BlockLocation {
  int corrupt;           // If the block is corrupt
  int numOfNodes;        // Number of Datanodes which keep the block
  char **hosts;          // Datanode hostnames
  char **names;          // Datanode IP:xferPort for accessing the block
  char **topologyPaths;  // Full path name in network topology
  tOffset length;        // block length, may be 0 for the last block
  tOffset offset;        // Offset of the block in the file
} BlockLocation;

/**
 * Get an array containing hostnames, offset and size of portions of the given file.
 *
 * @param fs The file system
 * @param path The path to the file
 * @param start The start offset into the given file
 * @param length The length for which to get locations for
 * @param numOfBlock Output the number of elements in the returned array
 *
 * @return An array of BlockLocation struct.
 */
BlockLocation *hdfsGetFileBlockLocations(hdfsFS fs, const char *path, tOffset start, tOffset length,
                                         int *numOfBlock);

/**
 * Free the BlockLocation array returned by hdfsGetFileBlockLocations
 *
 * @param locations The array returned by hdfsGetFileBlockLocations
 * @param numOfBlock The number of elements in the locaitons
 */
void hdfsFreeFileBlockLocations(BlockLocation *locations, int numOfBlock);

/**
 * Create encryption zone for the directory with specific key name
 * @param fs The configured filesystem handle.
 * @param path The path of the directory.
 * @param keyname The key name of the encryption zone
 * @return Returns 0 on success, -1 on error.
 */
int hdfsCreateEncryptionZone(hdfsFS fs, const char *path, const char *keyName);

/**
 * hdfsEncryptionZoneInfo - Get information about a path as a (dynamically
 * allocated) single hdfsEncryptionZoneInfo struct. hdfsEncryptionZoneInfo should be
 * called when the pointer is no longer needed.
 * @param fs The configured filesystem handle.
 * @param path The path of the encryption zone.
 * @return Returns a dynamically-allocated hdfsEncryptionZoneInfo object;
 * NULL on error.
 */
hdfsEncryptionZoneInfo *hdfsGetEZForPath(hdfsFS fs, const char *path);

/**
 * hdfsEncryptionZoneInfo -  Get list of all the encryption zones.
 * hdfsFreeEncryptionZoneInfo should be called to deallocate memory.
 * @param fs The configured filesystem handle.
 * @return Returns a dynamically-allocated array of hdfsEncryptionZoneInfo objects;
 * NULL on error.
 */
hdfsEncryptionZoneInfo *hdfsListEncryptionZones(hdfsFS fs, int *numEntries);

#ifdef __cplusplus
}
#endif

#endif /* _HDFS_LIBHDFS3_CLIENT_HDFS_H_ */
