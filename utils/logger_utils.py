import os
from typing import Optional
from loguru import logger


class LoggerManager:
    """
    Logger manager class for configuring and managing loguru logger
    This is a globally shared logger manager, loguru's logger is a global singleton, 
    configured once and available to all modules
    """
    
    # Class variable: track configured log file paths to avoid duplicate handler addition
    _configured_files: set = set()
    _default_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    def __init__(
        self,
        output_path: str,
        log_file_name: str = "log.log",
        level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "7 days",
        log_format: Optional[str] = None,
        console_output: bool = True,
    ):
        """
        Initialize Logger manager
        
        Args:
            output_path: Directory path where log files are saved
            log_file_name: Log file name, default is "log.log"
            level: Log level, options: "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", default is "INFO"
            rotation: Log file rotation size, e.g., "10 MB", "100 MB", default is "10 MB"
            retention: Log file retention time, e.g., "7 days", "1 month", default is "7 days"
            log_format: Log format string, default is colored format
            console_output: Whether to also output to console, default is True
        """
        self.output_path = output_path
        self.log_file_name = log_file_name
        self.level = level
        self.rotation = rotation
        self.retention = retention
        self.log_format = log_format or self._default_format
        self.console_output = console_output
        self.log_file_path: Optional[str] = None
    
    def setup(self, force_reconfigure: bool = False):
        """
        Configure logger, save log output to specified path
        
        Args:
            force_reconfigure: Whether to force reconfigure (even if already configured), default is False
        
        Returns:
            logger: Configured loguru logger instance (globally shared)
        """
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Build complete log file path (use absolute path to avoid duplicates)
        self.log_file_path = os.path.abspath(os.path.join(self.output_path, self.log_file_name))
        
        # Check if this file path has already been configured
        if self.log_file_path in self._configured_files and not force_reconfigure:
            logger.debug(f"Logger already configured for file {self.log_file_path}, skipping duplicate configuration")
            return logger
        
        # If console output is not needed, remove default console handler
        if not self.console_output:
            logger.remove()
        
        # Add file output handler
        logger.add(
            self.log_file_path,
            level=self.level,
            rotation=self.rotation,
            retention=self.retention,
            format=self.log_format,
            encoding="utf-8",
            enqueue=True,  # Async write, improve performance
        )
        
        # Record configured file path
        self._configured_files.add(self.log_file_path)
        
        logger.info(f"Logger configured, logs will be saved to: {self.log_file_path}")
        logger.info(f"Log level: {self.level}, File rotation: {self.rotation}, Retention: {self.retention}")
        
        return logger
    
    @classmethod
    def is_configured(cls, file_path: str) -> bool:
        """
        Check if specified file path has been configured
        
        Args:
            file_path: Log file path
        
        Returns:
            bool: Whether it is configured
        """
        abs_path = os.path.abspath(file_path)
        return abs_path in cls._configured_files
    
    @classmethod
    def get_configured_files(cls) -> set:
        """
        Get all configured log file paths
        
        Returns:
            set: Set of configured file paths
        """
        return cls._configured_files.copy()
    
    def __repr__(self) -> str:
        """Return string representation of object"""
        return (
            f"LoggerManager(output_path='{self.output_path}', "
            f"log_file_name='{self.log_file_name}', level='{self.level}')"
        )


def setup_logger(
    output_path: str,
    log_file_name: str = "record.log",
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "7 days",
    log_format: Optional[str] = None,
    console_output: bool = True,
    force_reconfigure: bool = False,
):
    """
    Configure loguru logger, save log output to specified path (convenience function interface)
    This is a globally shared logger, loguru's logger is a global singleton, 
    configured once and available to all modules
    
    Usage:
        1. Call setup_logger once in main.py for configuration
        2. Other modules can directly use `from loguru import logger`, no need to setup again
    
    Args:
        output_path: Directory path where log files are saved
        log_file_name: Log file name, default is "log.log"
        level: Log level, options: "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", default is "INFO"
        rotation: Log file rotation size, e.g., "10 MB", "100 MB", default is "10 MB"
        retention: Log file retention time, e.g., "7 days", "1 month", default is "7 days"
        log_format: Log format string, default is colored format
        console_output: Whether to also output to console, default is True
        force_reconfigure: Whether to force reconfigure (even if already configured), default is False
    
    Returns:
        logger: Configured loguru logger instance (globally shared)
        
    Example:
        # Method 1: Use convenience function (recommended)
        >>> from utils.x_utils import setup_logger
        >>> setup_logger("./output", "app.log", level="DEBUG")
        
        # Method 2: Use LoggerManager class
        >>> from utils.x_utils import LoggerManager
        >>> manager = LoggerManager("./output", "app.log", level="DEBUG")
        >>> manager.setup()
        
        # In other modules (e.g., module_a.py), all loguru methods can be used:
        >>> from loguru import logger
        >>> logger.trace("Trace information")      # Most detailed log
        >>> logger.debug("Debug information")      # Debug log
        >>> logger.info("General information")     # Info log
        >>> logger.success("Success information")   # Success log (with green marker)
        >>> logger.warning("Warning information")    # Warning log (with yellow marker)
        >>> logger.error("Error information")      # Error log (with red marker)
        >>> logger.critical("Critical error")      # Critical error log
        >>> logger.exception("Exception information")  # Exception log (automatically includes stack trace)
        # All logs will automatically be written to configured file and console
    """
    manager = LoggerManager(
        output_path=output_path,
        log_file_name=log_file_name,
        level=level,
        rotation=rotation,
        retention=retention,
        log_format=log_format,
        console_output=console_output,
    )
    return manager.setup(force_reconfigure=force_reconfigure)


