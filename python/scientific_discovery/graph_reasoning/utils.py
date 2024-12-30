from pathlib import Path
from typing import Union, List, Optional
import re
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarkdownConfig:
    """Configuration for markdown processing."""
    preserve_links: bool = False
    preserve_images: bool = False
    preserve_code_blocks: bool = False
    preserve_lists: bool = False
    extra_patterns: Optional[List[tuple[str, str]]] = None

class TextUtils:
    """Utility class for text processing operations."""
    
    @staticmethod
    def contains_phrase(text: str, phrase: str, case_sensitive: bool = True) -> bool:
        """
        Check if a phrase exists within a text string.
        
        Args:
            text: The main text to search in
            phrase: The phrase to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            bool: True if phrase is found in text
        """
        if not case_sensitive:
            text = text.lower()
            phrase = phrase.lower()
        return phrase in text

    @staticmethod
    def remove_markdown(
        text: str,
        config: Optional[MarkdownConfig] = None
    ) -> str:
        """
        Remove markdown formatting from text with configurable options.
        
        Args:
            text: The markdown text to process
            config: Configuration options for markdown processing
            
        Returns:
            str: Clean text without markdown formatting
        """
        if config is None:
            config = MarkdownConfig()

        patterns = [
            # Basic inline formatting
            (r'\*\*([^*]+)\*\*', r'\1'),  # Bold
            (r'\*([^*]+)\*', r'\1'),      # Italic
            (r'__([^_]+)__', r'\1'),      # Bold
            (r'_([^_]+)_', r'\1'),        # Italic
            (r'~~(.*?)~~', r'\1'),        # Strikethrough
            (r'`([^`]+)`', r'\1'),        # Inline code
            
            # Headers
            (r'#+\s', ''),
            
            # Blockquotes
            (r'^>\s+', '', re.MULTILINE),
        ]
        
        # Conditional patterns based on config
        if not config.preserve_links:
            patterns.extend([
                (r'\[([^\]]+)\]\([^\)]+\)', r'\1'),  # Links
            ])
            
        if not config.preserve_images:
            patterns.extend([
                (r'!\[[^\]]*\]\([^\)]+\)', ''),  # Images
            ])
            
        if not config.preserve_code_blocks:
            patterns.extend([
                (r'```.*?```', '', re.DOTALL),  # Code blocks
            ])
            
        if not config.preserve_lists:
            patterns.extend([
                (r'^[\*\-\+]\s+', '', re.MULTILINE),  # Unordered lists
                (r'^\d+\.\s+', '', re.MULTILINE),     # Ordered lists
            ])

        # Add any extra patterns from config
        if config.extra_patterns:
            patterns.extend(config.extra_patterns)

        # Apply all patterns
        result = text
        for pattern in patterns:
            if len(pattern) == 2:
                pattern_str, replacement = pattern
                flags = 0
            else:
                pattern_str, replacement, flags = pattern
            result = re.sub(pattern_str, replacement, result, flags=flags)

        # Clean up extra whitespace
        result = re.sub(r'\n\s*\n', '\n\n', result)
        return result.strip()

class FileUtils:
    """Utility class for file and directory operations."""
    
    @staticmethod
    def ensure_directory(
        path: Union[str, Path],
        mode: int = 0o755,
        parents: bool = True
    ) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path to ensure
            mode: Permission mode for new directory
            parents: Whether to create parent directories
            
        Returns:
            Path: Path object pointing to the ensured directory
            
        Raises:
            OSError: If directory creation fails
        """
        try:
            path = Path(path)
            if not path.exists():
                path.mkdir(mode=mode, parents=parents, exist_ok=True)
                logger.info(f"Created directory: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to ensure directory {path}: {str(e)}")
            raise

    @staticmethod
    def safe_filename(filename: str) -> str:
        """
        Convert a string to a safe filename.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Safe filename
        """
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        return filename[:255]

# Example usage:
"""
# Text utilities
text_utils = TextUtils()
has_phrase = text_utils.contains_phrase("Hello world", "world", case_sensitive=False)
print(f"Contains phrase: {has_phrase}")

# Markdown processing with custom config
config = MarkdownConfig(
    preserve_links=True,
    preserve_code_blocks=True,
    extra_patterns=[
        (r'custom_pattern', 'replacement'),
    ]
)
clean_text = text_utils.remove_markdown("**Bold** and _italic_", config)
print(f"Clean text: {clean_text}")

# File utilities
file_utils = FileUtils()
path = file_utils.ensure_directory("./data/processed")
print(f"Directory ensured: {path}")

safe_name = file_utils.safe_filename("My File: Version 1.0")
print(f"Safe filename: {safe_name}")
"""