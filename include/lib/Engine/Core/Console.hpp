#pragma once
#include "ISingleton.hpp"
namespace EvoEngine
{
    class Console : public ISingleton<Console>
    {
    public:
        /**
         * Push log to the console
         * @param msg The message to print.
         */
        static void Log(const std::string& msg);
        /**
         * Push error to the console
         * @param msg The message to print.
         */
        static void Error(const std::string& msg);
        /**
         * Push warning to the console
         * @param msg The message to print.
         */
        static void Warning(const std::string& msg);
    };

} // namespace EvoEngine

/**
 * \brief A thread-safe message log macro.
 * \param msg The log message
 */
#define EVOENGINE_LOG(msg)                                                                                             \
    {                                                                                                                  \
        EvoEngine::Console::Log(msg);                                                                                    \
        std::cout << "[EvoEngine]Log: " << msg << " (" << __FILE__ << ": line " << __LINE__ << ")\n==========" << std::endl;       \
    }

 /**
  * \brief A thread-safe error log macro.
  * \param msg The error message
  */
#define EVOENGINE_ERROR(msg)                                                                                           \
    {                                                                                                                  \
        EvoEngine::Console::Error(msg);                                                                                  \
        std::cerr << "[EvoEngine]Error: " << msg << " (" << __FILE__ << ": line " << __LINE__ << ")\n==========" << std::endl;     \
    }
  /**
   * \brief A thread-safe warning log macro.
   * \param msg The warning message
   */
#define EVOENGINE_WARNING(msg)                                                                                         \
    {                                                                                                                  \
        EvoEngine::Console::Warning(msg);                                                                                \
        std::cout << "[EvoEngine]Warning: " << msg << " (" << __FILE__ << ": line " << __LINE__ << ")\n==========" << std::endl;   \
    }