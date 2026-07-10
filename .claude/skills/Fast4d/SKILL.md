```markdown
# Fast4d Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the Fast4d TypeScript codebase. You'll learn about file naming, import/export styles, commit message conventions, and how to structure and run tests. This guide is ideal for contributors aiming to maintain consistency and quality in Fast4d projects.

## Coding Conventions

### File Naming
- **Pattern:** PascalCase
- **Example:**  
  ```plaintext
  MyComponent.ts
  UserService.ts
  ```

### Import Style
- **Pattern:** Relative imports
- **Example:**
  ```typescript
  import { UserService } from './UserService';
  import { Helper } from '../utils/Helper';
  ```

### Export Style
- **Pattern:** Named exports
- **Example:**
  ```typescript
  // In MyComponent.ts
  export function MyComponent() { ... }

  // In UserService.ts
  export const UserService = { ... };
  ```

### Commit Messages
- **Pattern:** Conventional commits with prefixes (e.g., `docs`)
- **Example:**
  ```
  docs: update README with installation instructions
  ```

## Workflows

### Documenting Changes
**Trigger:** When updating documentation or code comments  
**Command:** `/docs-update`

1. Make your documentation changes in relevant files.
2. Use a conventional commit message starting with `docs:`, e.g., `docs: add API usage section`.
3. Push your changes and open a pull request.

### Adding New Code
**Trigger:** When creating new modules, components, or utilities  
**Command:** `/add-code`

1. Name your new file using PascalCase (e.g., `NewFeature.ts`).
2. Use relative imports for dependencies.
3. Export your functions or objects using named exports.
4. Write or update corresponding test files as `NewFeature.test.ts`.
5. Commit using a relevant conventional prefix.

### Writing Tests
**Trigger:** When adding or updating tests  
**Command:** `/write-test`

1. Create a test file following the pattern: `*.test.ts` (e.g., `UserService.test.ts`).
2. Place test files alongside the code or in a dedicated test directory.
3. Write tests according to the chosen (unknown) framework's conventions.
4. Run tests to ensure they pass before committing.

## Testing Patterns

- **File Pattern:** All test files are named using the `*.test.ts` pattern.
- **Framework:** Not explicitly detected; follow standard TypeScript testing practices.
- **Example:**
  ```typescript
  // UserService.test.ts
  import { UserService } from './UserService';

  describe('UserService', () => {
    it('should return user data', () => {
      // test implementation
    });
  });
  ```

## Commands
| Command        | Purpose                                           |
|----------------|--------------------------------------------------|
| /docs-update   | Document changes or update code comments         |
| /add-code      | Add new modules, components, or utilities        |
| /write-test    | Add or update test files                         |
```
